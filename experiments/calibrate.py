"""Fit and measure confidence calibration for one judge.

    python -m experiments.calibrate --nli-model models/nli-scifact
    python -m experiments.calibrate --nli-model facebook/bart-large-mnli --fit-claims 300

What it does, in order:
  1. For the first --fit-claims TRAIN claims, judge each of the top-k BM25 papers and keep the
     raw logits. Each (claim, paper) pair has a known answer: entailment or contradiction if
     the annotators marked that paper as evidence, neutral otherwise.
  2. Fit one temperature T on those pairs (verification/calibration.py). Dev is never used.
  3. Judge the same kind of pairs for every DEV claim and report, before and after T:
       paper level: accuracy, NLL, ECE and the reliability bins;
       claim level: the pipeline's verdict confidence vs. whether the verdict was right
                    (UNKNOWN verdicts carry confidence 0 and are left out).
  4. Save experiments/results/calibration_<model>.json and record T in models/calibration.json,
     which the app reads so every verdict also shows a calibrated confidence.

Logits are cached in experiments/results/nli_logits_<model>_<split>.jsonl, so re-running
is instant and the fitted T is reproducible from files in the repo.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from data.loader import load_claims, load_config, load_corpus
from data.schema import Claim, RawLabel
from retrieval.bm25 import BM25Retriever
from verification.aggregate import DocumentVerdict, aggregate
from verification.calibration import expected_calibration_error, fit_temperature, reliability_table, summarize
from verification.nli import get_nli_logits_fn, prediction_from_logits
from verification.pipeline import calibrated_confidence

RAW_TO_NLI = {RawLabel.SUPPORT: "entailment", RawLabel.CONTRADICT: "contradiction"}


def pairs_for(claims: List[Claim], corpus, retriever, top_k: int) -> List[dict]:
    out = []
    for c in claims:
        for sd in retriever.retrieve(c.claim, top_k=top_k):
            doc = corpus[sd.doc_id]
            label = "neutral"
            rationales = c.evidence.get(str(sd.doc_id))
            if rationales:
                labels = {r.label for r in rationales}
                if len(labels) != 1:
                    continue
                label = RAW_TO_NLI[labels.pop()]
            out.append({"claim_id": c.id, "doc_id": sd.doc_id, "claim": c.claim,
                        "text": doc.title + " " + " ".join(doc.abstract), "label": label})
    return out


def logits_with_cache(pairs: List[dict], logits_fn, cache_path: Path) -> np.ndarray:
    cached: Dict[tuple, list] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cached[(row["claim_id"], row["doc_id"])] = row["logits"]
    missing = [p for p in pairs if (p["claim_id"], p["doc_id"]) not in cached]
    if missing:
        print(f"  judging {len(missing)} pairs ({len(pairs) - len(missing)} cached)...")
        t0 = time.perf_counter()
        with open(cache_path, "a", encoding="utf-8") as f:
            for i, p in enumerate(missing, start=1):
                lg = logits_fn(p["claim"], p["text"]).tolist()
                cached[(p["claim_id"], p["doc_id"])] = lg
                f.write(json.dumps({"claim_id": p["claim_id"], "doc_id": p["doc_id"], "label": p["label"],
                                    "logits": lg}) + "\n")
                if i % 50 == 0 or i == len(missing):
                    el = time.perf_counter() - t0
                    print(f"    [{i}/{len(missing)}] {el:.0f}s, eta {el / i * (len(missing) - i):.0f}s", flush=True)
    return np.array([cached[(p["claim_id"], p["doc_id"])] for p in pairs], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-model", required=True)
    ap.add_argument("--fit-claims", type=int, default=300, help="how many TRAIN claims to fit T on")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    data_cfg = load_config("configs/data.yaml")
    rcfg = yaml.safe_load(open("configs/retrieval.yaml", encoding="utf-8"))
    vcfg = yaml.safe_load(open("configs/verification.yaml", encoding="utf-8"))
    label_map, min_conf = vcfg["nli"]["label_map"], vcfg["aggregation"]["min_confidence_for_verdict"]
    raw_dir = Path(data_cfg["paths"]["raw_dir"])
    corpus = load_corpus(raw_dir, data_cfg["paths"]["corpus_file"])
    retriever = BM25Retriever(corpus, rcfg["bm25"])
    train = load_claims(raw_dir, data_cfg["paths"]["claims_train_file"], has_labels=True)[: args.fit_claims]
    dev = load_claims(raw_dir, data_cfg["paths"]["claims_dev_file"], has_labels=True)

    tag = args.nli_model.rstrip("/").split("/")[-1]
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading judge {args.nli_model}...")
    logits_fn, id2label = get_nli_logits_fn(args.nli_model)
    label2id = {v: k for k, v in id2label.items()}

    print(f"Train: {len(train)} claims")
    train_pairs = pairs_for(train, corpus, retriever, args.top_k)
    train_logits = logits_with_cache(train_pairs, logits_fn, results_dir / f"nli_logits_{tag}_train.jsonl")
    train_labels = [label2id[p["label"]] for p in train_pairs]
    temperature = fit_temperature(train_logits, train_labels)
    print(f"  fitted temperature T = {temperature:.3f} on {len(train_pairs)} pairs")

    print(f"Dev: {len(dev)} claims")
    dev_pairs = pairs_for(dev, corpus, retriever, args.top_k)
    dev_logits = logits_with_cache(dev_pairs, logits_fn, results_dir / f"nli_logits_{tag}_dev.jsonl")
    dev_labels = [label2id[p["label"]] for p in dev_pairs]
    paper_before = summarize(dev_logits, dev_labels, 1.0, args.bins)
    paper_after = summarize(dev_logits, dev_labels, temperature, args.bins)

    # Claim level: run the deployed aggregation rule on raw scores; only the reported confidence changes.
    by_claim: Dict[int, List[int]] = {}
    for i, p in enumerate(dev_pairs):
        by_claim.setdefault(p["claim_id"], []).append(i)
    conf_raw, conf_cal, correct, verdicts = [], [], [], {}
    for c in dev:
        idx = by_claim.get(c.id, [])
        per_doc = [DocumentVerdict(doc_id=dev_pairs[i]["doc_id"],
                                   prediction=prediction_from_logits(dev_logits[i], id2label, label_map, temperature))
                   for i in idx]
        agg = aggregate(per_doc, min_confidence_for_verdict=min_conf)
        verdicts[agg.verdict.value] = verdicts.get(agg.verdict.value, 0) + 1
        cal = calibrated_confidence(agg.verdict.value, per_doc, min_conf)
        if agg.verdict.value == "UNKNOWN" or cal is None:
            continue
        conf_raw.append(agg.confidence)
        conf_cal.append(cal)
        correct.append(agg.verdict == c.gold_verdict())
    claim_level = {
        "n_decided_claims": len(correct), "verdict_counts": verdicts,
        "accuracy_of_decided": round(float(np.mean(correct)), 4) if correct else None,
        "before": {"ece": round(expected_calibration_error(conf_raw, correct, args.bins), 4),
                   "mean_confidence": round(float(np.mean(conf_raw)), 4) if conf_raw else None,
                   "bins": reliability_table(conf_raw, correct, args.bins)},
        "after": {"ece": round(expected_calibration_error(conf_cal, correct, args.bins), 4),
                  "mean_confidence": round(float(np.mean(conf_cal)), 4) if conf_cal else None,
                  "bins": reliability_table(conf_cal, correct, args.bins)},
    }

    report = {
        "nli_model": args.nli_model, "temperature": round(temperature, 3),
        "fitted_on": {"split": "train", "claims": len(train), "pairs": len(train_pairs), "top_k": args.top_k},
        "paper_level_dev": {"pairs": len(dev_pairs), "before": paper_before, "after": paper_after},
        "claim_level_dev": claim_level,
        "_provenance": "Produced by experiments/calibrate.py on the user's machine. T fitted on train only.",
    }
    (results_dir / f"calibration_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    cal_file = Path(vcfg["demo"].get("calibration_file", "models/calibration.json"))
    cal_file.parent.mkdir(parents=True, exist_ok=True)
    registry = json.loads(cal_file.read_text(encoding="utf-8")) if cal_file.exists() else {}
    registry[args.nli_model] = {"temperature": round(temperature, 3), "fitted_on_train_claims": len(train),
                                "pairs": len(train_pairs)}
    cal_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(f"\npaper level (dev, {len(dev_pairs)} pairs): ECE {paper_before['ece']} -> {paper_after['ece']}, "
          f"NLL {paper_before['nll']} -> {paper_after['nll']}, accuracy {paper_before['accuracy']}")
    print(f"claim level (dev, {len(correct)} decided claims): ECE {claim_level['before']['ece']} -> "
          f"{claim_level['after']['ece']}, mean confidence {claim_level['before']['mean_confidence']} -> "
          f"{claim_level['after']['mean_confidence']}, accuracy {claim_level['accuracy_of_decided']}")
    print(f"written {results_dir / f'calibration_{tag}.json'} and {cal_file}")


if __name__ == "__main__":
    main()
