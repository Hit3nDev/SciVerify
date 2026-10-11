"""Run the proposed system (or any ablation of it) on the SciFact dev claims.

Examples:
    python -m experiments.run_proposed --nli-model facebook/bart-large-mnli
    python -m experiments.run_proposed --nli-model facebook/bart-large-mnli --no-sentence-selection
    python -m experiments.run_proposed --retrieval bm25 --limit 30      # quick smoke test

FAIR COMPARISON: Baseline C was run with facebook/bart-large-mnli. To compare
against it, pass --nli-model facebook/bart-large-mnli here too; otherwise a
difference could come from the model rather than from the new components.

Writes experiments/results/proposed_<retrieval>_<sel|nosel>_<model>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from data.loader import load_claims, load_config, load_corpus
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.verification_metrics import evaluate_verification
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.embeddings import get_embedder
from retrieval.hybrid import HybridRetriever
from verification.aggregate import DocumentVerdict, aggregate
from verification.evidence_selection import select_sentences
from verification.nli import get_nli_fn


def build_retriever(kind: str, corpus, rcfg: dict):
    if kind == "bm25":
        return BM25Retriever(corpus, rcfg["bm25"])
    embed_fn = get_embedder(rcfg["dense"]["model_name"])
    dense = DenseRetriever(corpus, rcfg["dense"], embed_fn)
    if kind == "dense":
        return dense
    if kind == "hybrid":
        return HybridRetriever(
            [BM25Retriever(corpus, rcfg["bm25"]), dense],
            rrf_k=rcfg["hybrid"]["rrf_k"], top_k=rcfg["bm25"]["top_k"],
        )
    raise ValueError(f"unknown retrieval '{kind}'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", choices=["bm25", "dense", "hybrid"])
    ap.add_argument("--no-sentence-selection", action="store_true")
    ap.add_argument("--nli-model")
    ap.add_argument("--sentences-per-doc", type=int, help="override proposed.sentences_per_doc")
    ap.add_argument("--limit", type=int, help="only the first N dev claims (smoke test)")
    args = ap.parse_args()

    data_cfg = load_config("configs/data.yaml")
    rcfg = yaml.safe_load(open("configs/retrieval.yaml", encoding="utf-8"))
    vcfg = yaml.safe_load(open("configs/verification.yaml", encoding="utf-8"))
    pcfg, ncfg = vcfg["proposed"], vcfg["nli"]

    retrieval_kind = args.retrieval or pcfg["retrieval"]
    use_sel = pcfg["sentence_selection"] and not args.no_sentence_selection
    nli_model = args.nli_model or ncfg["model_name"]
    n_sent = args.sentences_per_doc or pcfg["sentences_per_doc"]
    min_conf = vcfg["aggregation"]["min_confidence_for_verdict"]

    raw_dir = Path(data_cfg["paths"]["raw_dir"])
    corpus = load_corpus(raw_dir, data_cfg["paths"]["corpus_file"])
    dev = load_claims(raw_dir, data_cfg["paths"]["claims_dev_file"], has_labels=True)
    if args.limit:
        dev = dev[: args.limit]
    print(f"retrieval={retrieval_kind} sentence_selection={use_sel} nli={nli_model} claims={len(dev)}")

    retriever = build_retriever(retrieval_kind, corpus, rcfg)
    nli_fn = get_nli_fn(nli_model, ncfg["label_map"])

    gold, pred = [], []
    t0 = time.perf_counter()
    for i, claim in enumerate(dev, start=1):
        docs = retriever.retrieve(claim.claim, top_k=pcfg["evidence_top_k"])
        per_doc = []
        for sd in docs:
            doc = corpus[sd.doc_id]
            if use_sel:
                text, _ = select_sentences(claim.claim, doc, n_sent)
            else:
                text = doc.title + " " + " ".join(doc.abstract)
            per_doc.append(DocumentVerdict(doc_id=sd.doc_id, prediction=nli_fn(claim.claim, text)))
        gold.append(claim.gold_verdict())
        pred.append(aggregate(per_doc, min_confidence_for_verdict=min_conf).verdict)
        if i % 10 == 0 or i == len(dev):
            el = time.perf_counter() - t0
            print(f"  [{i}/{len(dev)}] {el:.0f}s elapsed, eta {el / i * (len(dev) - i):.0f}s", flush=True)
    total = time.perf_counter() - t0

    ver = evaluate_verification(gold, pred)
    ret = evaluate_retrieval(
        dev, lambda t, k: [s.doc_id for s in retriever.retrieve(t, top_k=k)],
        k_values=rcfg["evaluation"]["recall_at_k"])

    out = {
        "system": "proposed" if (retrieval_kind == "hybrid" and use_sel) else "ablation",
        "retrieval": retrieval_kind, "sentence_selection": use_sel,
        "sentences_per_doc": n_sent if use_sel else None, "nli_model": nli_model,
        "num_claims": len(dev),
        "retrieval_metrics": {"recall_at_k": ret.recall_at_k, "mrr": ret.mrr},
        "verification": {
            "accuracy": ver.accuracy, "macro_f1": ver.macro_f1,
            "macro_f1_gold_classes": ver.macro_f1_gold_classes,
            "per_class_precision": ver.per_class_precision,
            "per_class_recall": ver.per_class_recall,
            "per_class_f1": ver.per_class_f1, "support": ver.support},
        "total_time_seconds": round(total, 1),
        "avg_latency_per_claim_ms": round(total / len(dev) * 1000, 1),
    }
    tag = f"proposed_{retrieval_kind}_{('sel' + (str(n_sent) if args.sentences_per_doc else '')) if use_sel else 'nosel'}_{nli_model.split('/')[-1]}"
    if args.limit:
        tag += f"_limit{args.limit}"
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    path = Path("experiments/results") / f"{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten {path}\n{json.dumps(out, indent=2)}")


if __name__ == "__main__":
    main()
