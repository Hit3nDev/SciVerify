"""Tests for the research additions: calibration maths, the counter-claim rules, the
fine-tuning example builder, the logits->prediction function and calibrated confidence.
All pure functions; no model or dataset is needed."""

import numpy as np
import pytest

from data.schema import Claim, CorpusDocument, Rationale, RawLabel
from experiments.finetune_nli import build_examples, split_by_claim
from retrieval.bm25 import ScoredDoc
from verification.calibration import (expected_calibration_error, fit_temperature, negative_log_likelihood,
                                      reliability_table, softmax, summarize)
from verification.counterclaim import build_counterclaim, judge_consistency
from verification.nli import NliPrediction, prediction_from_logits
from verification.pipeline import ClaimVerifier, calibrated_confidence
from verification.aggregate import DocumentVerdict

LABEL_MAP = {"entailment": "SUPPORT", "contradiction": "CONTRADICT", "neutral": "NEI"}


# ------------------------------------------------------------------ calibration
def overconfident_data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, n)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), labels] += 1.2
    return logits * 3, labels          # x3: same argmax, far too sure


def test_temperature_lowers_nll_and_ece_without_changing_argmax():
    logits, labels = overconfident_data()
    T = fit_temperature(logits, labels)
    assert T > 1.5
    assert negative_log_likelihood(logits, labels, T) < negative_log_likelihood(logits, labels, 1.0)
    before, after = summarize(logits, labels, 1.0), summarize(logits, labels, T)
    assert after["ece"] < before["ece"] / 3
    assert after["accuracy"] == before["accuracy"]
    assert np.array_equal(softmax(logits).argmax(1), softmax(logits, T).argmax(1))


def test_fit_temperature_near_one_for_already_calibrated_logits():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, 3000)
    logits = rng.normal(0, 1, (3000, 3)) + np.eye(3)[labels] * 1.0
    # sample labels from the model's own softmax so it is calibrated by construction
    probs = softmax(logits)
    labels = np.array([rng.choice(3, p=p) for p in probs])
    assert abs(fit_temperature(logits, labels) - 1.0) < 0.15


def test_reliability_table_and_ece():
    conf = [0.95, 0.92, 0.55, 0.52, 0.3]
    ok = [True, True, False, True, False]
    rows = reliability_table(conf, ok, n_bins=10)
    assert sum(r["count"] for r in rows) == 5
    assert rows[9]["count"] == 2 and rows[9]["accuracy"] == 1.0 and rows[9]["confidence"] == pytest.approx(0.935)
    assert rows[8]["count"] == 0 and rows[8]["accuracy"] is None
    assert rows[5]["count"] == 2 and rows[5]["accuracy"] == 0.5
    assert expected_calibration_error([1.0, 1.0, 0.5, 0.5], [True, True, True, False]) == 0.0
    assert expected_calibration_error([1.0, 1.0], [False, False]) == 1.0


def test_fit_temperature_rejects_empty():
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((0, 3)), [])


# ------------------------------------------------------------------ nli
def test_prediction_from_logits_follows_label_names_not_positions():
    bart = {0: "contradiction", 1: "neutral", 2: "entailment"}
    minilm = {0: "contradiction", 1: "entailment", 2: "neutral"}
    p = prediction_from_logits(np.array([0.0, 0.0, 4.0]), bart, LABEL_MAP)
    assert p.raw_nli_label == "entailment" and p.mapped_label == RawLabel.SUPPORT and p.calibrated_score is None
    q = prediction_from_logits(np.array([0.0, 4.0, 0.0]), minilm, LABEL_MAP)
    assert q.raw_nli_label == "entailment" and q.mapped_label == RawLabel.SUPPORT
    n = prediction_from_logits(np.array([0.0, 0.0, 4.0]), minilm, LABEL_MAP)
    assert n.mapped_label is None and n.probs["neutral"] == pytest.approx(n.score)


def test_prediction_calibrated_score_is_softer_for_temperature_above_one():
    p = prediction_from_logits(np.array([0.0, 0.0, 4.0]), {0: "contradiction", 1: "neutral", 2: "entailment"},
                               LABEL_MAP, temperature=2.0)
    assert p.calibrated_score is not None and p.calibrated_score < p.score
    assert p.raw_nli_label == "entailment"


# ------------------------------------------------------------------ counter-claims
@pytest.mark.parametrize("claim, expected, method", [
    ("Febrile seizures increase the threshold for epilepsy.", "Febrile seizures decrease the threshold for epilepsy.", "antonym"),
    ("Low nucleosome occupancy correlates with low methylation.", "High nucleosome occupancy correlates with low methylation.", "antonym"),
    ("Albendazole is used to treat lymphatic filariasis.", "Albendazole is not used to treat lymphatic filariasis.", "negate"),
    ("Teaching hospitals do not provide better care than others.", "Teaching hospitals do provide better care than others.", "remove-negation"),
    ("Aspirin has no effect on PGE2.", "Aspirin has a effect on PGE2.", "remove-negation"),
    ("0-dimensional biomaterials show inductive properties.", "0-dimensional biomaterials do not show inductive properties.", "negate"),
    ("ITAM phosphorylation prevents the transfer of the signal.", "ITAM phosphorylation does not prevent the transfer of the signal.", "negate"),
    ("Albendazole carries a risk.", "Albendazole does not carry a risk.", "negate"),
    ("The activation of monocytes has deleterious effects.", "The activation of monocytes does not have deleterious effects.", "negate"),
    ("Vitamin D has been shown to protect bone.", "Vitamin D has not been shown to protect bone.", "negate"),
    ("Aspirin PGE2.", "It is not the case that aspirin PGE2.", "prefix"),
])
def test_counterclaim_rules(claim, expected, method):
    cc = build_counterclaim(claim)
    assert (cc.text, cc.method) == (expected, method)


def test_counterclaim_reports_what_changed_and_rejects_empty():
    cc = build_counterclaim("X reduces Y.")
    assert cc.changed == ("reduces", "increases")
    with pytest.raises(ValueError):
        build_counterclaim("   ")


@pytest.mark.parametrize("a, b, status", [
    ("SUPPORTED", "CONTRADICTED", "consistent"), ("CONTRADICTED", "SUPPORTED", "consistent"),
    ("SUPPORTED", "SUPPORTED", "inconsistent"), ("CONTRADICTED", "CONTRADICTED", "inconsistent"),
    ("SUPPORTED", "UNKNOWN", "inconclusive"), ("UNKNOWN", "UNKNOWN", "inconclusive"),
    ("SUPPORTED", "CONFLICTING", "inconclusive"),
])
def test_judge_consistency(a, b, status):
    assert judge_consistency(a, b)[0] == status


# ------------------------------------------------------------------ fine-tuning data
CORPUS = {
    1: CorpusDocument(1, "Paper one", ["s0", "s1", "s2"]),
    2: CorpusDocument(2, "Paper two", ["t0", "t1"]),
    3: CorpusDocument(3, "Paper three", ["u0"]),
    4: CorpusDocument(4, "Paper four", ["v0"]),
    5: CorpusDocument(5, "Paper five", ["w0"]),
}


class Retriever:
    def retrieve(self, text, top_k=3):
        return [ScoredDoc(d, 1.0) for d in (1, 4, 5)][:top_k]


def test_build_examples_kinds_labels_and_negative_cap():
    claims = [
        Claim(10, "claim A", {"1": [Rationale([1, 2], RawLabel.SUPPORT)]}, cited_doc_ids=[1, 2]),
        Claim(11, "claim B", {"3": [Rationale([0], RawLabel.CONTRADICT), Rationale([0], RawLabel.SUPPORT)]},
              cited_doc_ids=[3]),
        Claim(12, "claim C", {}, cited_doc_ids=[2]),
    ]
    ex = build_examples(claims, CORPUS, Retriever(), negatives_per_claim=2)
    a = [e for e in ex if e["claim_id"] == 10]
    assert [(e["kind"], e["label"]) for e in a] == [("abstract", "entailment"), ("rationale", "entailment"),
                                                    ("neutral", "neutral"), ("neutral", "neutral")]
    assert a[1]["text"] == "Paper one s1 s2"
    assert [e["doc_id"] for e in a if e["kind"] == "neutral"] == [2, 4]   # cited-no-evidence first, then retrieved
    assert all(e["label"] == "neutral" for e in ex if e["claim_id"] == 11)   # mixed-label paper skipped
    c = [e for e in ex if e["claim_id"] == 12]
    assert [e["doc_id"] for e in c] == [2, 1]


def test_build_examples_ignores_out_of_range_rationale_indices():
    claims = [Claim(1, "c", {"3": [Rationale([0, 7], RawLabel.SUPPORT)]}, cited_doc_ids=[])]
    ex = build_examples(claims, CORPUS, Retriever(), negatives_per_claim=0)
    assert [e["kind"] for e in ex] == ["abstract", "rationale"] and ex[1]["text"] == "Paper three u0"


def test_split_by_claim_keeps_a_claim_on_one_side():
    rows = [{"claim_id": i // 3} for i in range(30)]
    train, val = split_by_claim(rows, 0.2, seed=1)
    assert len(train) + len(val) == 30 and len(val) == 6
    assert not ({r["claim_id"] for r in train} & {r["claim_id"] for r in val})


# ------------------------------------------------------------------ calibrated confidence
def dv(doc_id, label, score, cal):
    return DocumentVerdict(doc_id, NliPrediction("x", label, score, calibrated_score=cal))


def test_calibrated_confidence_follows_the_aggregation_rule():
    docs = [dv(1, RawLabel.SUPPORT, 0.9, 0.7), dv(2, RawLabel.SUPPORT, 0.3, 0.2), dv(3, None, 0.9, 0.8)]
    assert calibrated_confidence("SUPPORTED", docs, 0.5) == pytest.approx(0.7)   # 0.3 is below threshold
    docs.append(dv(4, RawLabel.CONTRADICT, 0.8, 0.6))
    assert calibrated_confidence("CONFLICTING", docs, 0.5) == pytest.approx(0.6)
    assert calibrated_confidence("UNKNOWN", docs, 0.5) is None
    assert calibrated_confidence("SUPPORTED", [dv(1, RawLabel.SUPPORT, 0.9, None)], 0.5) is None


def test_pipeline_reports_calibrated_confidence_and_judge_name():
    corpus = {1: CorpusDocument(1, "Exercise and memory", ["Exercise improved memory."])}

    def nli(claim, text):
        return NliPrediction("entailment", RawLabel.SUPPORT, 0.95, {"entailment": 0.95}, calibrated_score=0.72)

    v = ClaimVerifier(corpus, Retriever(), nli, judge_name="fake judge")
    v.corpus = corpus
    r = ClaimVerifier({1: corpus[1], 4: corpus[1], 5: corpus[1]}, Retriever(), nli, judge_name="fake judge").verify("Exercise helps memory")
    assert r.confidence == pytest.approx(0.95) and r.confidence_calibrated == pytest.approx(0.72)
    assert r.judge == "fake judge" and r.evidence[0].calibrated_score == 0.72
    assert r.to_dict()["confidence_calibrated"] == 0.72
