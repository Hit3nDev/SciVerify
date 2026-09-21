"""Tests for verification/aggregate.py and evaluation/verification_metrics.py.

No real NLI model is needed here -- we construct NliPrediction objects
directly to test the aggregation rules (spec section 12: conflicting
evidence must not be silently resolved to one side).
"""

from __future__ import annotations

import pytest

from data.schema import RawLabel, Verdict
from evaluation.verification_metrics import evaluate_verification
from verification.aggregate import DocumentVerdict, aggregate
from verification.nli import NliPrediction


def _pred(mapped_label, score):
    raw = {RawLabel.SUPPORT: "entailment", RawLabel.CONTRADICT: "contradiction", None: "neutral"}[
        mapped_label
    ]
    return NliPrediction(raw_nli_label=raw, mapped_label=mapped_label, score=score)


def test_aggregate_no_documents_returns_unknown():
    result = aggregate([])
    assert result.verdict == Verdict.UNKNOWN
    assert result.confidence == 0.0


def test_aggregate_single_support_document():
    docs = [DocumentVerdict(doc_id=1, prediction=_pred(RawLabel.SUPPORT, 0.9))]
    result = aggregate(docs)
    assert result.verdict == Verdict.SUPPORTED
    assert result.confidence == pytest.approx(0.9)


def test_aggregate_single_contradict_document():
    docs = [DocumentVerdict(doc_id=1, prediction=_pred(RawLabel.CONTRADICT, 0.85))]
    result = aggregate(docs)
    assert result.verdict == Verdict.CONTRADICTED


def test_aggregate_conflicting_evidence_is_not_silently_resolved():
    docs = [
        DocumentVerdict(doc_id=1, prediction=_pred(RawLabel.SUPPORT, 0.8)),
        DocumentVerdict(doc_id=2, prediction=_pred(RawLabel.CONTRADICT, 0.75)),
    ]
    result = aggregate(docs)
    assert result.verdict == Verdict.CONFLICTING
    assert "mixed" in result.explanation.lower() or "conflict" in result.explanation.lower()


def test_aggregate_below_confidence_threshold_is_unknown():
    docs = [DocumentVerdict(doc_id=1, prediction=_pred(RawLabel.SUPPORT, 0.3))]
    result = aggregate(docs, min_confidence_for_verdict=0.5)
    assert result.verdict == Verdict.UNKNOWN


def test_aggregate_all_neutral_is_unknown():
    docs = [
        DocumentVerdict(doc_id=1, prediction=_pred(None, 0.9)),
        DocumentVerdict(doc_id=2, prediction=_pred(None, 0.7)),
    ]
    result = aggregate(docs)
    assert result.verdict == Verdict.UNKNOWN


def test_aggregate_majority_support_ignores_weak_contradict_below_threshold():
    docs = [
        DocumentVerdict(doc_id=1, prediction=_pred(RawLabel.SUPPORT, 0.9)),
        DocumentVerdict(doc_id=2, prediction=_pred(RawLabel.CONTRADICT, 0.2)),  # below threshold
    ]
    result = aggregate(docs, min_confidence_for_verdict=0.5)
    assert result.verdict == Verdict.SUPPORTED  # weak contradict doesn't trigger conflict


def test_verification_metrics_perfect_predictions():
    gold = [Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.UNKNOWN]
    pred = [Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.UNKNOWN]
    result = evaluate_verification(gold, pred)
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0


def test_verification_metrics_all_wrong():
    gold = [Verdict.SUPPORTED, Verdict.SUPPORTED]
    pred = [Verdict.CONTRADICTED, Verdict.CONTRADICTED]
    result = evaluate_verification(gold, pred)
    assert result.accuracy == 0.0
    assert result.macro_f1 == 0.0


def test_verification_metrics_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        evaluate_verification([Verdict.SUPPORTED], [Verdict.SUPPORTED, Verdict.UNKNOWN])


def test_verification_metrics_empty_raises():
    with pytest.raises(ValueError):
        evaluate_verification([], [])


def test_gold_verdict_matches_aggregate_vocabulary():
    """Sanity check that Claim.gold_verdict() and aggregate() produce
    comparable Verdict values for direct accuracy/F1 comparison."""
    from data.schema import Claim, Rationale

    conflicting_claim = Claim(
        id=1,
        claim="x",
        evidence={
            "10": [Rationale(sentences=[0], label=RawLabel.SUPPORT)],
            "20": [Rationale(sentences=[1], label=RawLabel.CONTRADICT)],
        },
        cited_doc_ids=[10, 20],
    )
    assert conflicting_claim.gold_verdict() == Verdict.CONFLICTING
