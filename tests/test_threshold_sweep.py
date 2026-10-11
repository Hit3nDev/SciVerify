"""Tests for experiments/sweep_threshold.py using hand-built saved predictions."""
from data.schema import Verdict
from experiments.sweep_threshold import best_threshold, reaggregate, score, sweep


def _row(gold, label, nli, s):
    return {"claim_id": 1, "claim": "x", "gold": gold, "pred": "",
            "confidence": s, "docs": [{"doc_id": 1, "nli": nli, "label": label, "score": s}]}


def test_threshold_changes_verdict():
    rows = [_row("UNKNOWN", "SUPPORT", "entailment", 0.6)]
    assert reaggregate(rows, 0.5) == [Verdict.SUPPORTED]
    assert reaggregate(rows, 0.9) == [Verdict.UNKNOWN]


def test_neutral_doc_is_unknown_at_any_threshold():
    rows = [_row("UNKNOWN", None, "neutral", 0.95)]
    assert reaggregate(rows, 0.4) == [Verdict.UNKNOWN]


def test_best_threshold_picks_the_one_that_scores_best():
    # NEI claim with a weak entailment (0.6) plus a real support with a strong one (0.95):
    rows = [_row("UNKNOWN", "SUPPORT", "entailment", 0.6),
            _row("SUPPORTED", "SUPPORT", "entailment", 0.95)]
    results = sweep(rows, [0.5, 0.8])
    assert score(rows, 0.8)["accuracy"] == 1.0
    assert best_threshold(results) == 0.8


def test_ties_go_to_lower_threshold():
    rows = [_row("SUPPORTED", "SUPPORT", "entailment", 0.99)]
    assert best_threshold(sweep(rows, [0.5, 0.9])) == 0.5
