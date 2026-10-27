"""Tests for the Telegram bot: message text (pure functions), the async API client against the real
FastAPI app with fake models, and that the bot wires up without contacting Telegram."""

import asyncio

import httpx
import pytest

from api.main import create_app
from bots.telegram_bot import (SciVerifyClient, ServerDown, BadClaim, bar, build_application, claim_problem,
                               claim_tips, counts, fit, format_counter, format_papers, format_progress,
                               format_result, format_why, gold_line, public_web_url, sureness)
from data.schema import RawLabel
from tests.test_pipeline_api_ui import CORPUS, FakeRetriever
from tests.test_web_app import JUDGES, LABELLED, honest_nli
from verification.nli import NliPrediction
from verification.pipeline import ClaimVerifier


def ev(doc_id, judgement, score, title="Exercise & <memory>", passage="Exercise improved memory."):
    return {"doc_id": doc_id, "title": title, "judgement": judgement, "nli_score": score, "passage": passage,
            "retrieval_score": 5.0}


def result(verdict, evidence, confidence=0.9, calibrated=None, **kw):
    return {"claim": "Exercise improves <memory>", "verdict": verdict, "confidence": confidence,
            "confidence_calibrated": calibrated, "evidence": evidence, "settings": {"top_k": 3, "min_confidence": 0.5},
            "judge": "General judge (bart-large-mnli)", "judge_key": "a", "elapsed_seconds": 2.34, **kw}


# ------------------------------------------------------------------ plain-language text
def test_supported_result_is_friendly_escaped_and_uses_calibrated_confidence():
    r = result("SUPPORTED", [ev(1, "SUPPORT", 0.97), ev(2, "NEUTRAL", 0.8), ev(3, "SUPPORT", 0.3)], 0.97, 0.78)
    text = format_result(r, corpus_size=5183)
    assert "SUPPORTED BY RESEARCH" in text and "back up" in text
    assert "&lt;memory&gt;" in text and "<memory>" not in text            # user text can't break the HTML
    assert "78%, fairly sure" in text and "raw score was 97%" in text
    assert text.count("✔ <b>Agrees</b>") == 1                              # the 0.3 paper is too weak to show
    assert "plus 2 more" in text and "5,183" in text and "not medical advice" in text


def test_unknown_result_explains_it_is_not_false_and_lists_closest_papers():
    text = format_result(result("UNKNOWN", [ev(1, "NEUTRAL", 0.9)], 0.0))
    assert "does <b>not</b> mean the claim is false" in text
    assert "Closest papers" in text and "How sure" not in text


def test_conflicting_and_empty_and_tips_and_cached():
    text = format_result(result("CONFLICTING", [ev(1, "SUPPORT", 0.9), ev(2, "CONTRADICT", 0.8)], cached=True),
                         tips=["send one claim"])
    assert text.startswith("<b>Tip:</b> send one claim")
    assert "1 paper(s) agree, 1 disagree" in text and "✘ <b>Disagrees</b>" in text and "instant" in text
    assert "no papers related" in format_result(result("UNKNOWN", [], 0.0))


def test_papers_why_and_counts():
    r = result("SUPPORTED", [ev(1, "SUPPORT", 0.97), ev(2, "CONTRADICT", 0.4), ev(3, "NEUTRAL", 0.9)])
    assert counts(r) == {"agree": 1, "disagree": 0, "uncounted": 2}
    papers = format_papers(r)
    assert "All 3 papers" in papers and "too unsure to count" in papers and "paper #2" in papers
    why = format_why(r)
    assert "1 agreed, 0 disagreed, 2 didn't count" in why and "kept the top 3" in why
    assert "General judge" in why and "bart" not in why and "<b>Supported</b>" in why and "2.3 seconds" in why
    assert format_papers(result("UNKNOWN", [])) == "I didn't find any papers for this claim."


def test_counter_message_explains_each_status():
    c = {"claim": "X raises Y", "verdict": "SUPPORTED", "counter_claim": "X lowers Y", "counter_verdict": "SUPPORTED",
         "method": "antonym", "changed": ["raises", "lowers"], "status": "inconsistent"}
    text = format_counter(c)
    assert "Failed" in text and "same answer" in text and "swapped “raises” for “lowers”" in text
    assert "Passed" in format_counter({**c, "counter_verdict": "CONTRADICTED", "status": "consistent"})
    assert "added “It is not the case that”" in format_counter({**c, "method": "prefix", "changed": None,
                                                              "status": "inconclusive"})


def test_progress_stages():
    assert "Step 1 of 2" in format_progress("c", "searching", corpus_size=5183) and "5,183" in format_progress("c", "searching", corpus_size=5183)
    assert "waking up" in format_progress("c", "loading")
    p = format_progress("c", "read", [{"title": "A"}, {"title": "B"}], 1)
    assert "Found 2 related" in p and "<code>1/2</code>" in p


def test_claim_checks_and_tips():
    assert "too short" not in (claim_problem("Aspirin reduces heart attacks") or "")
    assert claim_problem("Aspirin reduces heart attacks") is None
    assert "short" in claim_problem("aspirin?")
    assert "long" in claim_problem("word " * 300)
    assert "statements" in claim_tips("Does coffee cause cancer?")[0]
    assert any("one at a time" in t for t in claim_tips("A causes B. C causes D. E causes F. G causes H."))
    assert any("personal" in t for t in claim_tips("My headache is caused by sugar."))
    assert claim_tips("Smoking increases the risk of lung cancer.") == []


def test_small_helpers():
    assert bar(0.0) == "□" * 10 and bar(1.0) == "■" * 10 and bar(0.74) == "■" * 7 + "□" * 3
    assert [sureness(x) for x in (0.95, 0.8, 0.65, 0.3)] == ["very sure", "fairly sure", "somewhat sure", "not very sure"]
    assert public_web_url("http://127.0.0.1:8000") is None and public_web_url("http://localhost:8000/") is None
    assert public_web_url("https://sciverify.example.org/") == "https://sciverify.example.org"
    assert "right" in gold_line({"verdict": "SUPPORTED"}, "SUPPORTED") and "wrong" in gold_line({"verdict": "UNKNOWN"}, "SUPPORTED")


def test_quoted_passage_drops_the_repeated_title():
    r = result("SUPPORTED", [ev(1, "SUPPORT", 0.9, title="Smoking study", passage="Smoking study Smoking caused cancer.")])
    assert "“Smoking caused cancer.”" in format_result(r) and "“Smoking caused cancer.”" in format_papers(r)


def test_fit_keeps_long_messages_valid():
    text = "<b>" + "\n".join(["word " * 20] * 100) + "</b>"
    out = fit(text)
    assert len(out) <= 4096 and out.count("<b>") == out.count("</b>") and "shortened" in out


# ------------------------------------------------------------------ client against the real API (fake models)
def contradicting_nli(claim, text):
    return NliPrediction("contradiction", RawLabel.CONTRADICT, 0.85)


@pytest.fixture()
def api_client():
    def factory(key):
        return ClaimVerifier(CORPUS, FakeRetriever([1, 2]), honest_nli if key == "a" else contradicting_nli,
                             judge_name=f"Judge {key.upper()}")

    app = create_app(factory, lambda: LABELLED, lambda: {}, judges=JUDGES)
    return SciVerifyClient("http://test", transport=httpx.ASGITransport(app=app))


def test_client_streams_progress_and_returns_result(api_client):
    async def go():
        seen = []

        async def on_event(e):
            seen.append(e["event"])

        r = await api_client.verify_stream("Exercise improves memory", "b", on_event)
        await api_client.close()
        return seen, r

    seen, r = asyncio.run(go())
    assert seen[0] == "loading" and "retrieved" in seen and seen.count("judged") == 2
    assert r["verdict"] == "CONTRADICTED" and r["judge_key"] == "b"
    assert "CONTRADICTED BY RESEARCH" in format_result(r)


def test_client_other_endpoints_and_errors(api_client):
    async def go():
        ex = await api_client.example(label="SUPPORTED")
        judges = await api_client.judges()
        c = await api_client.counterclaim("Exercise improves memory", "a")
        with pytest.raises(BadClaim):
            await api_client.verify("   ")
        with pytest.raises(BadClaim):
            await api_client.verify_stream("   ", None, lambda e: None)
        await api_client.close()
        return ex, judges, c

    ex, judges, c = asyncio.run(go())
    assert ex["gold_verdict"] == "SUPPORTED"
    assert [j["key"] for j in judges["judges"]] == ["a", "b"]
    assert c["status"] == "consistent" and "Passed" in format_counter(c)


def test_client_reports_server_down():
    async def go():
        client = SciVerifyClient("http://127.0.0.1:9")            # nothing listens on port 9
        try:
            with pytest.raises(ServerDown):
                await client.status()
            with pytest.raises(ServerDown):
                await client.verify_stream("Exercise improves memory", None, lambda e: None)
        finally:
            await client.close()

    asyncio.run(go())


def test_bot_builds_offline_with_all_handlers():
    app = build_application("123456:TEST-token", "http://127.0.0.1:8000")
    handlers = [h for group in app.handlers.values() for h in group]
    commands = {c for h in handlers for c in getattr(h, "commands", ())}
    assert {"start", "help", "example", "quiz", "judge", "verdicts", "how", "about"} <= commands
    assert len(handlers) >= 12
