"""Classification metrics for claim verdicts (SUPPORTED / CONTRADICTED /
UNKNOWN / CONFLICTING), using scikit-learn under the hood so results match
standard, widely-trusted implementations rather than a custom formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from sklearn.metrics import precision_recall_fscore_support

from data.schema import Verdict


@dataclass
class VerificationEvalResult:
    accuracy: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    macro_f1: float
    support: Dict[str, int]  # number of gold examples per class
    labels_present: List[str]


def evaluate_verification(
    gold: List[Verdict], predicted: List[Verdict]
) -> VerificationEvalResult:
    if len(gold) != len(predicted):
        raise ValueError(
            f"gold ({len(gold)}) and predicted ({len(predicted)}) must be the same length"
        )
    if not gold:
        raise ValueError("evaluate_verification called with no examples")

    labels_present = sorted({v.value for v in gold} | {v.value for v in predicted})
    gold_str = [v.value for v in gold]
    pred_str = [v.value for v in predicted]

    correct = sum(1 for g, p in zip(gold_str, pred_str) if g == p)
    accuracy = correct / len(gold_str)

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        gold_str, pred_str, labels=labels_present, zero_division=0
    )

    macro_f1 = float(sum(f1s) / len(f1s)) if len(f1s) else 0.0

    return VerificationEvalResult(
        accuracy=accuracy,
        per_class_precision=dict(zip(labels_present, (float(p) for p in precisions))),
        per_class_recall=dict(zip(labels_present, (float(r) for r in recalls))),
        per_class_f1=dict(zip(labels_present, (float(f) for f in f1s))),
        macro_f1=macro_f1,
        support=dict(zip(labels_present, (int(s) for s in supports))),
        labels_present=labels_present,
    )
