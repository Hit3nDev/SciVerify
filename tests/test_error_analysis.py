"""Tests for experiments/error_analysis.py using hand-built prediction rows."""
from experiments.error_analysis import (categorize_error, claim_flags, confusion_matrix,
                                        error_rate_by_flag)


def _row(gold, pred, doc_ids=(1, 2, 3), claim="a claim"):
    return {"claim_id": 1, "claim": claim, "gold": gold, "pred": pred,
            "docs": [{"doc_id": d, "nli": "neutral", "label": None, "score": 0.9} for d in doc_ids]}


def test_correct_prediction_has_no_category():
    assert categorize_error(_row("SUPPORTED", "SUPPORTED"), [1]) is None


def test_verdict_without_gold_evidence():
    assert categorize_error(_row("UNKNOWN", "SUPPORTED"), []) == "verdict_without_gold_evidence"


def test_retrieval_miss_when_gold_doc_not_retrieved():
    assert categorize_error(_row("SUPPORTED", "UNKNOWN", doc_ids=(7, 8, 9)), [1]) == "retrieval_miss"


def test_evidence_retrieved_not_recognised():
    assert categorize_error(_row("SUPPORTED", "UNKNOWN"), [1]) == "evidence_retrieved_not_recognised"


def test_polarity_flip():
    assert categorize_error(_row("SUPPORTED", "CONTRADICTED"), [1]) == "polarity_flip"


def test_spurious_conflict_and_missed_conflict():
    assert categorize_error(_row("SUPPORTED", "CONFLICTING"), [1]) == "spurious_conflict"
    assert categorize_error(_row("CONFLICTING", "SUPPORTED"), [1, 2]) == "gold_conflict_missed"


def test_retrieval_miss_takes_priority_over_polarity():
    # gold doc not retrieved, so the wrong polarity cannot be blamed on the NLI model
    assert categorize_error(_row("SUPPORTED", "CONTRADICTED", doc_ids=(7,)), [1]) == "retrieval_miss"


def test_flags():
    f = claim_flags("Drug X cut risk by 40%", [1, 2], median_words=3)
    assert f == {"numeric_claim": True, "multi_evidence_claim": True, "long_claim": True}
    g = claim_flags("Exercise helps", [1], median_words=3)
    assert g == {"numeric_claim": False, "multi_evidence_claim": False, "long_claim": False}


def test_error_rate_by_flag_and_confusion_matrix():
    rows = [_row("SUPPORTED", "SUPPORTED"), _row("SUPPORTED", "UNKNOWN")]
    flags = [{"numeric_claim": True}, {"numeric_claim": False}]
    stats = error_rate_by_flag(rows, flags)
    assert stats["numeric_claim=True"]["error_rate"] == 0.0
    assert stats["numeric_claim=False"]["error_rate"] == 1.0
    assert confusion_matrix(rows)["SUPPORTED"] == {"SUPPORTED": 1, "UNKNOWN": 1}
