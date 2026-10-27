"""Tests for the web-app features: options, streaming, sentence analysis, search, examples,
benchmarks, caching and the static page. Same fakes as test_pipeline_api_ui.py, no model needed."""

import json

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from data.schema import RawLabel
from tests.test_pipeline_api_ui import CORPUS, FakeRetriever, verifier
from verification.nli import NliPrediction
from verification.pipeline import ClaimVerifier, claim_terms

LABELLED = [{"id": 1, "claim": "Exercise improves memory in older adults", "gold_verdict": "SUPPORTED",
             "gold_doc_ids": [1]},
            {"id": 2, "claim": "Sleep has no effect", "gold_verdict": "UNKNOWN", "gold_doc_ids": []}]


@pytest.fixture()
def client(tmp_path):
    (tmp_path / "bm25_baseline.json").write_text(json.dumps({"recall_at_k": {"10": 0.9}}))
    (tmp_path / "broken.json").write_text("{not json")
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return verifier([1, 2, 3])

    c = TestClient(create_app(factory, lambda: LABELLED, lambda: {"nli_model": "fake"}, results_dir=tmp_path))
    c.factory_calls = calls
    return c


def ndjson(text):
    return [json.loads(line) for line in text.splitlines() if line]


# ------------------------------------------------------------------ pipeline
def test_verify_steps_order_and_payloads():
    events = list(verifier([1, 3]).verify_steps("Exercise improves memory"))
    assert [k for k, _ in events] == ["retrieved", "judged", "judged", "done"]
    assert [p["doc_id"] for p in events[0][1]] == [1, 3]
    assert events[-1][1].verdict == "SUPPORTED"


def test_top_k_and_threshold_options():
    v = verifier([1, 2, 3])
    assert len(v.verify("Exercise memory", top_k=1).evidence) == 1
    assert v.verify("Exercise memory", top_k=2).verdict == "CONFLICTING"
    # the contradicting paper scores 0.90: with a 0.92 threshold it no longer counts
    assert v.verify("Exercise memory", top_k=2, min_confidence=0.92).verdict == "SUPPORTED"
    assert v.verify("Exercise memory", top_k=99, min_confidence=5).settings == {"top_k": 10, "min_confidence": 1.0}


def test_evidence_carries_full_abstract_and_highlights():
    e = verifier([1]).verify("Exercise improved memory scores").evidence[0]
    assert e.sentences == CORPUS[1].abstract
    assert e.highlighted and all(0 <= i < len(e.sentences) for i in e.highlighted)


def test_claim_terms_drop_stopwords_and_duplicates():
    assert claim_terms("The effect of IL 6 on memory in 5 mice and the Memory") == ["effect", "6", "memory", "5", "mice"]


def test_explain_sentences_judges_each_sentence():
    seen = []

    def nli(claim, text):
        seen.append(text)
        return NliPrediction("neutral", None, 0.7, {"entailment": .1, "neutral": .7, "contradiction": .2})

    v = ClaimVerifier(CORPUS, FakeRetriever([1]), nli)
    out = v.explain_sentences("claim", 1)
    assert [s["index"] for s in out] == [0, 1] and seen == CORPUS[1].abstract
    assert out[0]["judgement"] == "NEUTRAL" and out[0]["probabilities"]["neutral"] == 0.7
    with pytest.raises(KeyError):
        v.explain_sentences("claim", 999)


def test_search_returns_snippets_without_calling_the_model():
    def boom(*a):
        raise AssertionError("search must not use the model")

    hits = ClaimVerifier(CORPUS, FakeRetriever([1, 3]), boom).search("memory")
    assert [h["doc_id"] for h in hits] == [1, 3] and "memory" in hits[0]["snippet"].lower()
    assert ClaimVerifier(CORPUS, FakeRetriever([1]), boom).search("   ") == []


# ------------------------------------------------------------------ API
def test_verify_accepts_options_and_attaches_gold(client):
    body = client.post("/verify", json={"claim": "Exercise improves memory in older adults", "top_k": 1}).json()
    assert len(body["evidence"]) == 1 and body["settings"]["top_k"] == 1
    assert body["gold"] == {"gold_verdict": "SUPPORTED", "gold_doc_ids": [1]}
    assert client.post("/verify", json={"claim": "Exercise improves memory"}).json()["gold"] is None


@pytest.mark.parametrize("payload", [{"claim": "x", "top_k": 0}, {"claim": "x", "top_k": 11},
                                     {"claim": "x", "min_confidence": 1.5}])
def test_verify_rejects_bad_options(client, payload):
    assert client.post("/verify", json=payload).status_code == 422


def test_repeated_claim_is_served_from_cache(client):
    first = client.post("/verify", json={"claim": "Exercise improves memory"}).json()
    second = client.post("/verify", json={"claim": "Exercise improves memory "}).json()
    assert first["cached"] is False and second["cached"] is True
    assert second["verdict"] == first["verdict"]


def test_stream_emits_stages_then_result(client):
    r = client.post("/verify/stream", json={"claim": "Exercise improves memory"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    events = ndjson(r.text)
    kinds = [e["event"] for e in events]
    assert kinds[-1] == "done" and kinds.count("judged") == 3 and "retrieved" in kinds
    assert events[-1]["result"]["verdict"] == "CONFLICTING"
    again = ndjson(client.post("/verify/stream", json={"claim": "Exercise improves memory"}).text)
    assert again[-1]["result"]["cached"] is True and [e["event"] for e in again].count("judged") == 3


def test_stream_rejects_blank_claim(client):
    assert client.post("/verify/stream", json={"claim": "   "}).status_code == 422


def test_stream_reports_errors_inside_the_stream():
    def broken():
        raise RuntimeError("model download failed")

    c = TestClient(create_app(broken, lambda: [], lambda: {}))
    events = ndjson(c.post("/verify/stream", json={"claim": "x"}).text)
    assert events[-1] == {"event": "error", "detail": "model download failed"}
    assert c.get("/status").json()["error"] == "model download failed"


def test_explain_endpoint():
    def nli(claim, sentence):
        return NliPrediction("entailment", RawLabel.SUPPORT, 0.8) if "memory" in sentence \
            else NliPrediction("neutral", None, 0.9)

    client = TestClient(create_app(lambda: ClaimVerifier(CORPUS, FakeRetriever([1]), nli), lambda: [], lambda: {}))
    out = client.post("/explain", json={"claim": "Exercise improves memory", "doc_id": 1}).json()["sentences"]
    assert [s["judgement"] for s in out] == ["NEUTRAL", "SUPPORT"]
    assert client.post("/explain", json={"claim": "x", "doc_id": 999}).status_code == 404
    assert client.post("/explain", json={"claim": "  ", "doc_id": 1}).status_code == 422


def test_search_and_paper_endpoints(client):
    assert [r["doc_id"] for r in client.get("/search", params={"q": "memory", "k": 2}).json()["results"]] == [1, 2]
    assert client.get("/papers/3").json()["sentences"] == CORPUS[3].abstract
    assert client.get("/papers/999").status_code == 404
    assert client.get("/search", params={"q": ""}).status_code == 422


def test_examples_sample_and_filter(client):
    assert len(client.get("/examples", params={"n": 5}).json()["claims"]) == 2
    only = client.get("/examples", params={"n": 5, "label": "unknown"}).json()
    assert [c["id"] for c in only["claims"]] == [2] and only["pool_size"] == 1


def test_examples_without_dataset_are_empty():
    def missing():
        raise FileNotFoundError("no data")

    c = TestClient(create_app(lambda: verifier([1]), missing, lambda: {}))
    assert c.get("/examples").json() == {"claims": [], "pool_size": 0}


def test_benchmarks_reads_result_files_and_skips_broken_ones(client):
    assert client.get("/benchmarks").json() == {"bm25_baseline": {"recall_at_k": {"10": 0.9}}}


def test_status_and_warmup_load_the_model_once(client):
    s = client.get("/status").json()
    assert s["model_ready"] is False and s["nli_model"] == "fake"
    client.post("/warmup")
    client.post("/verify", json={"claim": "Exercise memory"})
    client.post("/verify", json={"claim": "Exercise memory again"})
    s = client.get("/status").json()
    assert s["model_ready"] is True and s["corpus_size"] == 3
    assert client.factory_calls["n"] == 1


def test_web_page_and_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "SciVerify" in page.text
    for asset in ["/static/app.js", "/static/charts.js", "/static/styles.css"]:
        assert client.get(asset).status_code == 200


# ------------------------------------------------------------------ several judges, counter-claims
JUDGES = [{"key": "a", "path": "fake/a", "name": "Judge A", "description": "", "calibrated": False},
          {"key": "b", "path": "fake/b", "name": "Judge B", "description": "", "calibrated": True}]


def contrarian_nli(claim, text):
    """Judge B: says SUPPORT to everything that mentions memory, including the opposite claim."""
    return NliPrediction("entailment", RawLabel.SUPPORT, 0.9, {"entailment": 0.9}, calibrated_score=0.6)


def honest_nli(claim, text):
    """Judge A: supports 'improves', contradicts 'does not improve' / 'worsens'."""
    negated = "not" in claim or "worsen" in claim
    return (NliPrediction("contradiction", RawLabel.CONTRADICT, 0.8) if negated
            else NliPrediction("entailment", RawLabel.SUPPORT, 0.95))


@pytest.fixture()
def multi():
    built = []

    def factory(key):
        built.append(key)
        return ClaimVerifier(CORPUS, FakeRetriever([1]), honest_nli if key == "a" else contrarian_nli,
                             judge_name=f"Judge {key.upper()}")

    c = TestClient(create_app(factory, lambda: [], lambda: {}, judges=JUDGES))
    c.built = built
    return c


def test_judges_endpoint_and_model_selection(multi):
    j = multi.get("/judges").json()
    assert j["default"] == "a" and [x["key"] for x in j["judges"]] == ["a", "b"]
    assert all(x["ready"] is False for x in j["judges"])
    a = multi.post("/verify", json={"claim": "Exercise improves memory"}).json()
    b = multi.post("/verify", json={"claim": "Exercise improves memory", "model": "b"}).json()
    assert a["judge"] == "Judge A" and a["judge_key"] == "a" and a["confidence_calibrated"] is None
    assert b["judge"] == "Judge B" and b["confidence_calibrated"] == 0.6 and b["cached"] is False
    assert multi.built == ["a", "b"]
    assert [x["ready"] for x in multi.get("/judges").json()["judges"]] == [True, True]
    assert multi.post("/verify", json={"claim": "x", "model": "nope"}).status_code == 422


def test_cache_is_per_judge(multi):
    multi.post("/verify", json={"claim": "Exercise improves memory", "model": "a"})
    assert multi.post("/verify", json={"claim": "Exercise improves memory", "model": "b"}).json()["cached"] is False
    assert multi.post("/verify", json={"claim": "Exercise improves memory", "model": "a"}).json()["cached"] is True


def test_counterclaim_flags_a_judge_that_agrees_with_both_sides(multi):
    good = multi.post("/counterclaim", json={"claim": "Exercise improves memory", "model": "a"}).json()
    assert good["counter_claim"] == "Exercise worsens memory" and good["method"] == "antonym"
    assert (good["verdict"], good["counter_verdict"], good["status"]) == ("SUPPORTED", "CONTRADICTED", "consistent")
    bad = multi.post("/counterclaim", json={"claim": "Exercise improves memory", "model": "b"}).json()
    assert (bad["verdict"], bad["counter_verdict"], bad["status"]) == ("SUPPORTED", "SUPPORTED", "inconsistent")
    assert "do not trust" in bad["note"] and bad["counter_evidence"][0]["doc_id"] == 1
    assert multi.post("/counterclaim", json={"claim": "   "}).status_code == 422


def test_warmup_and_stream_accept_a_judge(multi):
    assert multi.post("/warmup", params={"model": "b"}).json()["judge"] == "b"
    events = ndjson(multi.post("/verify/stream", json={"claim": "Exercise improves memory", "model": "b"}).text)
    assert events[-1]["result"]["judge_key"] == "b"
    assert multi.post("/warmup", params={"model": "zzz"}).status_code == 422
