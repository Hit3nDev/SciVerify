"""Tune the aggregation confidence threshold on TRAIN predictions, then report on DEV.

Why train: choosing the threshold on the dev set and then reporting dev accuracy
would be tuning on the test data. The threshold is chosen on train claims only;
dev is used once, to report.

Workflow (the NLI model runs once per split; the sweep itself takes seconds):
    python -m experiments.run_proposed --retrieval bm25 --no-sentence-selection \\
        --nli-model facebook/bart-large-mnli --split train
    python -m experiments.run_proposed --retrieval bm25 --no-sentence-selection \\
        --nli-model facebook/bart-large-mnli --split dev
    python -m experiments.sweep_threshold \\
        --tune experiments/results/proposed_bm25_nosel_bart-large-mnli_train_predictions.jsonl \\
        --apply experiments/results/proposed_bm25_nosel_bart-large-mnli_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from data.schema import RawLabel, Verdict
from evaluation.verification_metrics import evaluate_verification
from verification.aggregate import DocumentVerdict, aggregate
from verification.nli import NliPrediction

DEFAULT_THRESHOLDS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98]


def load_predictions(path: str | Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def reaggregate(rows: List[dict], threshold: float) -> List[Verdict]:
    """Recompute each claim's verdict from the saved per-document NLI outputs."""
    verdicts = []
    for row in rows:
        docs = [
            DocumentVerdict(
                doc_id=d["doc_id"],
                prediction=NliPrediction(
                    raw_nli_label=d["nli"],
                    mapped_label=RawLabel(d["label"]) if d["label"] else None,
                    score=d["score"],
                ),
            )
            for d in row["docs"]
        ]
        verdicts.append(aggregate(docs, min_confidence_for_verdict=threshold).verdict)
    return verdicts


def score(rows: List[dict], threshold: float) -> Dict[str, float]:
    gold = [Verdict(r["gold"]) for r in rows]
    res = evaluate_verification(gold, reaggregate(rows, threshold))
    return {"accuracy": res.accuracy, "macro_f1_gold_classes": res.macro_f1_gold_classes}


def sweep(rows: List[dict], thresholds: List[float]) -> Dict[float, Dict[str, float]]:
    return {t: score(rows, t) for t in thresholds}


def best_threshold(results: Dict[float, Dict[str, float]]) -> float:
    """Highest macro-F1 over gold classes; ties go to higher accuracy, then lower threshold."""
    return min(results, key=lambda t: (-results[t]["macro_f1_gold_classes"], -results[t]["accuracy"], t))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", required=True, help="predictions .jsonl from a --split train run")
    ap.add_argument("--apply", required=True, help="predictions .jsonl from a --split dev run")
    args = ap.parse_args()

    train_rows, dev_rows = load_predictions(args.tune), load_predictions(args.apply)
    tuned = sweep(train_rows, DEFAULT_THRESHOLDS)
    chosen = best_threshold(tuned)

    print(f"{'threshold':>9} {'train acc':>10} {'train mF1':>10} {'dev acc':>8} {'dev mF1':>8}")
    dev_all = sweep(dev_rows, DEFAULT_THRESHOLDS)
    for t in DEFAULT_THRESHOLDS:
        mark = "  <- chosen on train" if t == chosen else ""
        print(f"{t:>9.2f} {tuned[t]['accuracy']:>10.3f} {tuned[t]['macro_f1_gold_classes']:>10.3f} "
              f"{dev_all[t]['accuracy']:>8.3f} {dev_all[t]['macro_f1_gold_classes']:>8.3f}{mark}")
    print(f"\nDev result at train-chosen threshold {chosen:.2f}: accuracy {dev_all[chosen]['accuracy']:.3f}, "
          f"macro-F1 {dev_all[chosen]['macro_f1_gold_classes']:.3f}")
    print(f"Dev result at default 0.50:            accuracy {dev_all[0.5]['accuracy']:.3f}, "
          f"macro-F1 {dev_all[0.5]['macro_f1_gold_classes']:.3f}")
    out = {"chosen_threshold": chosen, "train": {str(k): v for k, v in tuned.items()},
           "dev": {str(k): v for k, v in dev_all.items()}}
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/threshold_sweep.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
