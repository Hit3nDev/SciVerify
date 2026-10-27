"""Tests for verification/pipeline.py, api/main.py and frontend/app.py.

A fake retriever and fake NLI model are injected, so these tests need no dataset and no model
download. They verify the logic and the interfaces; they say nothing about real-world accuracy.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from data.schema import CorpusDocument, RawLabel
from retrieval.bm25 import ScoredDoc
from verification.nli import NliPrediction
from verification.pipeline import MAX_CLAIM_CHARS, ClaimVerifier

CORPUS = {
    1: CorpusDocument(1, "Exercise and memory", ["Participants exercised.", "Exercise improved memory scores in older adults."]),
    2: CorpusDocument(2, "Exercise harms memory?", ["A small trial.", "Exercise did not improve memory."]),
    3: CorpusDocument(3, "Sleep study", ["Teenagers sleep less.", "Sleep duration has fallen."]),
}


class FakeRetriever:
    def __init__(self, doc_ids, score=2.0):
        self.doc_ids, self.score = doc_ids, score

    def retrieve(self, text, top_k=3):
        return [ScoredDoc(d, self.score) for d in self.doc_ids][:top_k]


def fake_nli(labels_by_doc):
    """NLI stub: looks at the evidence title to decide what to answer."""
    table = {"Exercise and memory": ("entailment", RawLabel.SUPPORT, 0.95),
             "Exercise harms memory?": ("contradiction", RawLabel.CONTRADICT, 0.90),
             "Sleep study": ("neutral", None, 0.80)}

    def fn(claim, evidence_text):
        for title, (raw, mapped, score) in table.items():
            if evidence_text.startswith(title):
                return NliPrediction(raw_nli_label=raw, mapped_label=mapped, score=score)
        raise AssertionError("unexpected evidence text")
    return fn


def verifier(doc_ids, score=2.0):
    return ClaimVerifier(CORPUS, FakeRetriever(doc_ids, score), fake_nli(None))


# ------------------------------------------------------------------ pipeline
def test_supported_with_evidence_and_passage():
    r = verifier([1, 3]).verify("Exercise improves memory in older adults")
    assert r.verdict == "SUPPORTED"
    assert len(r.evidence) == 2
    assert "memory" in r.evidence[0].passage.lower()
    assert r.confidence == pytest.approx(0.95)


def test_conflicting_evidence_is_not_forced_to_one_side():
    r = verifier([1, 2]).verify("Exercise improves memory")
    assert r.verdict == "CONFLICTING"


def test_multiple_evidence_passages_all_reported():
    r = verifier([1, 2, 3]).verify("Exercise improves memory")
    assert [e.doc_id for e in r.evidence] == [1, 2, 3]
    assert all(e.passage for e in r.evidence)


def test_no_relevant_papers_is_unknown_without_calling_the_model():
    def boom(*a):
        raise AssertionError("NLI must not run when nothing was retrieved")

    v = ClaimVerifier(CORPUS, FakeRetriever([1, 2], score=0.0), boom)
    r = v.verify("completely unrelated words")
    assert r.verdict == "UNKNOWN" and r.evidence == [] and r.confidence == 0.0


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_empty_claim_raises(bad):
    with pytest.raises(ValueError):
        verifier([1]).verify(bad)


def test_extremely_long_claim_is_shortened_not_crashing():
    r = verifier([1]).verify("exercise memory " * 2000)
    assert len(r.claim) == MAX_CLAIM_CHARS
    assert "shortened" in r.explanation


def test_to_dict_has_the_specified_fields():
    d = verifier([1, 3]).verify("Exercise improves memory").to_dict()
    for key in ["verdict", "confidence", "evidence", "sources", "retrieval_scores", "verification_scores", "explanation"]:
        assert key in d
    assert d["sources"][0] == {"doc_id": 1, "title": "Exercise and memory"}


# ------------------------------------------------------------------ API
@pytest.fixture()
def client():
    return TestClient(create_app(lambda: verifier([1, 3])))


def test_api_verify_ok(client):
    r = client.post("/verify", json={"claim": "Exercise improves memory in older adults"})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "SUPPORTED"
    assert len(body["retrieval_scores"]) == len(body["verification_scores"]) == 2


def test_api_health(client):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("payload", [{}, {"claim": ""}, {"claim": "   "}, {"claim": 123}, {"claim": None}, {"wrong": "x"}])
def test_api_rejects_bad_requests(client, payload):
    assert client.post("/verify", json=payload).status_code == 422


def test_api_rejects_malformed_json(client):
    r = client.post("/verify", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_api_rejects_oversized_claim(client):
    assert client.post("/verify", json={"claim": "x" * 6000}).status_code == 422


# ------------------------------------------------------------------ Streamlit page
def test_streamlit_page_runs_and_shows_verdict(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from verification import factory

    st.cache_resource.clear()
    monkeypatch.setattr(factory, "build_verifier", lambda *a, **k: verifier([1, 3]))

    from pathlib import Path
    app_path = str(Path(__file__).resolve().parent.parent / "frontend" / "app.py")
    at = AppTest.from_file(app_path, default_timeout=20).run()
    assert not at.exception

    at.button[0].click().run()                       # empty claim -> warning, no crash
    assert any("enter a claim" in w.value for w in at.warning)

    at.text_area[0].set_value("Exercise improves memory in older adults")
    at.button[0].click().run()
    assert not at.exception
    assert any("SUPPORTED" in s.value for s in at.success)
