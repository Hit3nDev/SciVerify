"""Categorise the errors in a saved predictions file.

    python -m experiments.error_analysis \
        --predictions experiments/results/proposed_bm25_nosel_bart-large-mnli_predictions.jsonl \
        --split dev

Automatic categories (one per error, decided in this order):
  gold_conflict_missed              gold says the literature conflicts, system did not
  spurious_conflict                 system said CONFLICTING, gold does not
  verdict_without_gold_evidence     gold = UNKNOWN (no evidence exists) but system gave SUPPORTED/CONTRADICTED
  retrieval_miss                    gold evidence document was not among the documents given to the NLI model
  evidence_retrieved_not_recognised gold evidence document WAS retrieved, system still said UNKNOWN
  polarity_flip                     gold evidence retrieved, system said the opposite of gold

Orthogonal flags (error rate with vs without): numeric claim, claim needs >=2 evidence documents,
claim longer than the median.

These are objective proxies. Categories that need reading the text (terminology mismatch,
ambiguous claim, misleading evidence, near-duplicate irrelevant paper) cannot be detected
automatically, so every error is also written to a CSV with a blank `manual_category` column.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Set

from data.loader import load_claims, load_config

_DIGIT = re.compile(r"\d")
_WANT = {"SUPPORTED": "entailment", "CONTRADICTED": "contradiction"}  # NLI label that matches a gold verdict


def categorize_error(row: dict, gold_doc_ids: Iterable[int]) -> Optional[str]:
    """Return the error category for one prediction row, or None if it is correct."""
    gold, pred = row["gold"], row["pred"]
    if gold == pred:
        return None
    if gold == "CONFLICTING":
        return "gold_conflict_missed"
    if pred == "CONFLICTING":
        return "spurious_conflict"
    if gold == "UNKNOWN":
        return "verdict_without_gold_evidence"
    retrieved = {d["doc_id"] for d in row["docs"]}
    if not (set(gold_doc_ids) & retrieved):
        return "retrieval_miss"
    if pred == "UNKNOWN":
        return "evidence_retrieved_not_recognised"
    return "polarity_flip"


def judge_on_gold_documents(rows: List[dict], gold_ids_by_claim: Dict[int, Iterable[int]]) -> Dict[str, int]:
    """How often the NLI model labels the ANNOTATED evidence document correctly.

    Only claims whose gold verdict is SUPPORTED/CONTRADICTED are considered. This separates
    'the judge got the key paper wrong' from 'the judge was right but the final rule lost it'.
    """
    out: Counter = Counter()
    for r in rows:
        if r["gold"] not in _WANT:
            continue
        out["gold_evidence_claims"] += 1
        ids = set(gold_ids_by_claim[r["claim_id"]])
        docs = [d for d in r["docs"] if d["doc_id"] in ids]
        if not docs:
            out["gold_doc_not_retrieved"] += 1
            continue
        out["gold_doc_retrieved"] += 1
        if any(d["nli"] == _WANT[r["gold"]] for d in docs):
            out["judge_right"] += 1
            out["judge_right_final_right" if r["gold"] == r["pred"] else "judge_right_final_wrong"] += 1
        elif all(d["nli"] == "neutral" for d in docs):
            out["judge_said_neutral"] += 1
        else:
            out["judge_said_opposite"] += 1
    return dict(out)


def recombination_ceiling(rows: List[dict]) -> int:
    """Claims that COULD be answered correctly by some rule that only recombines the saved
    per-document NLI labels, assuming perfect knowledge of which claims have evidence and which
    document to trust. An optimistic upper bound for any decision rule, not an achievable score."""
    reachable = 0
    for r in rows:
        labels = {d["nli"] for d in r["docs"]}
        if r["gold"] == "UNKNOWN":
            reachable += 1
        elif r["gold"] == "CONFLICTING":
            reachable += {"entailment", "contradiction"} <= labels
        elif r["gold"] in _WANT:
            reachable += _WANT[r["gold"]] in labels
    return reachable


def confusion_matrix(rows: List[dict]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        matrix[r["gold"]][r["pred"]] += 1
    return {g: dict(p) for g, p in matrix.items()}


def claim_flags(claim_text: str, gold_doc_ids: Iterable[int], median_words: float) -> Dict[str, bool]:
    return {
        "numeric_claim": bool(_DIGIT.search(claim_text)),
        "multi_evidence_claim": len(set(gold_doc_ids)) >= 2,
        "long_claim": len(claim_text.split()) > median_words,
    }


def error_rate_by_flag(rows: List[dict], flags: List[Dict[str, bool]]) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    for name in flags[0]:
        for value in (True, False):
            subset = [r for r, f in zip(rows, flags) if f[name] == value]
            errors = sum(1 for r in subset if r["gold"] != r["pred"])
            result[f"{name}={value}"] = {
                "claims": len(subset), "errors": errors,
                "error_rate": errors / len(subset) if subset else None,
            }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--split", choices=["dev", "train"], default="dev")
    ap.add_argument("--examples", type=int, default=3, help="examples to print per category")
    args = ap.parse_args()

    cfg = load_config("configs/data.yaml")
    claims_file = cfg["paths"]["claims_dev_file" if args.split == "dev" else "claims_train_file"]
    claims = {c.id: c for c in load_claims(Path(cfg["paths"]["raw_dir"]), claims_file, has_labels=True)}
    rows = [json.loads(l) for l in open(args.predictions, encoding="utf-8") if l.strip()]

    med = median(len(r["claim"].split()) for r in rows)
    categories: Counter = Counter()
    by_category: Dict[str, List[dict]] = defaultdict(list)
    flags_all = []
    csv_rows = []
    for r in rows:
        gold_ids = claims[r["claim_id"]].gold_doc_ids()
        flags_all.append(claim_flags(r["claim"], gold_ids, med))
        cat = categorize_error(r, gold_ids)
        if cat:
            categories[cat] += 1
            by_category[cat].append(r)
            csv_rows.append({
                "claim_id": r["claim_id"], "claim": r["claim"], "gold": r["gold"], "pred": r["pred"],
                "auto_category": cat, "gold_doc_ids": " ".join(map(str, gold_ids)),
                "retrieved_doc_ids": " ".join(str(d["doc_id"]) for d in r["docs"]),
                "nli_per_doc": "; ".join(f"{d['doc_id']}:{d['nli']}({d['score']:.2f})" for d in r["docs"]),
                "manual_category": "",
            })

    total_err = sum(categories.values())
    print(f"{len(rows)} claims, {total_err} errors ({total_err / len(rows):.1%})\n")
    print("Confusion matrix (rows = gold, columns = predicted):")
    for g, preds in sorted(confusion_matrix(rows).items()):
        print(f"  {g:13s} " + "  ".join(f"{p}={n}" for p, n in sorted(preds.items())))
    print("\nErrors by automatic category:")
    for cat, n in categories.most_common():
        print(f"  {cat:36s} {n:4d}  ({n / total_err:.0%} of errors)")
    flag_stats = error_rate_by_flag(rows, flags_all)
    print("\nError rate by claim property:")
    for k, v in flag_stats.items():
        rate = "n/a" if v["error_rate"] is None else f"{v['error_rate']:.1%}"
        print(f"  {k:28s} claims={v['claims']:4d} errors={v['errors']:4d} rate={rate}")
    gold_ids_by_claim = {r["claim_id"]: claims[r["claim_id"]].gold_doc_ids() for r in rows}
    judge = judge_on_gold_documents(rows, gold_ids_by_claim)
    ceiling = recombination_ceiling(rows)
    print("\nJudge on the annotated evidence document:")
    for k, v in judge.items():
        print(f"  {k:28s} {v}")
    print(f"\nCeiling for any rule that only recombines the saved judgements: {ceiling}/{len(rows)} = {ceiling / len(rows):.3f}"
          " (optimistic: assumes perfect knowledge of which claims have evidence)")
    print("\nExamples:")
    for cat, items in by_category.items():
        print(f"\n[{cat}]")
        for r in items[: args.examples]:
            print(f"  #{r['claim_id']} gold={r['gold']} pred={r['pred']}: {r['claim']}")
            for d in r["docs"]:
                print(f"      doc {d['doc_id']}: {d['nli']} ({d['score']:.2f})")

    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "error_analysis.json").write_text(json.dumps({
        "claims": len(rows), "errors": total_err, "by_category": dict(categories),
        "confusion_matrix": confusion_matrix(rows), "by_claim_property": flag_stats,
        "judge_on_gold_documents": judge, "recombination_ceiling": {"correct": ceiling, "of": len(rows)}}, indent=2))
    with open(out_dir / "error_analysis_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]) if csv_rows else ["claim_id"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nwritten {out_dir / 'error_analysis.json'} and {out_dir / 'error_analysis_errors.csv'}")


if __name__ == "__main__":
    main()
