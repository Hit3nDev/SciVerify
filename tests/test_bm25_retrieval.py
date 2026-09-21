"""Tests for retrieval/bm25.py and evaluation/retrieval_metrics.py.

Uses a small synthetic corpus so results are deterministic and don't
depend on the real (network-blocked-in-sandbox) SciFact download.
"""

from __future__ import annotations

import pytest

from data.schema import Claim, CorpusDocument
from evaluation.retrieval_metrics import evaluate_retrieval, reciprocal_rank, recall_at_k
from retrieval.bm25 import BM25Retriever

BM25_CONFIG = {"tokenizer": "simple_lowercase_alnum", "k1": 1.5, "b": 0.75, "top_k": 3}


@pytest.fixture()
def corpus():
    return {
        1: CorpusDocument(
            doc_id=1,
            title="Exercise and cognition in older adults",
            abstract=[
                "Regular physical exercise is associated with improved cognitive performance.",
                "Older adults who exercised showed better memory test scores.",
            ],
        ),
        2: CorpusDocument(
            doc_id=2,
            title="Sleep patterns in adolescents",
            abstract=["Adolescent sleep duration has declined over the past decade."],
        ),
        3: CorpusDocument(
            doc_id=3,
            title="Diet and cardiovascular risk",
            abstract=["High sodium diets are linked to increased blood pressure."],
        ),
    }


def test_bm25_retrieves_relevant_doc_first(corpus):
    retriever = BM25Retriever(corpus, BM25_CONFIG)
    results = retriever.retrieve("Does exercise improve cognitive performance in older adults?")
    assert results[0].doc_id == 1  # the exercise/cognition abstract should rank highest


def test_bm25_respects_top_k(corpus):
    retriever = BM25Retriever(corpus, BM25_CONFIG)
    results = retriever.retrieve("exercise cognition", top_k=2)
    assert len(results) == 2


def test_recall_at_k_hit_and_miss():
    assert recall_at_k(gold_ids=[1], retrieved_ids=[3, 1, 2], k=2) == 1.0
    assert recall_at_k(gold_ids=[1], retrieved_ids=[3, 2, 1], k=2) == 0.0


def test_recall_at_k_raises_on_empty_gold():
    with pytest.raises(ValueError):
        recall_at_k(gold_ids=[], retrieved_ids=[1, 2], k=5)


def test_reciprocal_rank():
    assert reciprocal_rank(gold_ids=[2], retrieved_ids=[1, 2, 3]) == 0.5
    assert reciprocal_rank(gold_ids=[9], retrieved_ids=[1, 2, 3]) == 0.0


def test_evaluate_retrieval_end_to_end(corpus):
    claims = [
        Claim(
            id=1,
            claim="Exercise improves cognitive performance in older adults.",
            evidence={"1": []},  # non-empty evidence dict => has gold doc id 1
            cited_doc_ids=[1],
        ),
        Claim(
            id=2,
            claim="An unrelated claim with no evidence at all.",
            evidence={},  # NEI — must be excluded from scoring
            cited_doc_ids=[],
        ),
    ]
    retriever = BM25Retriever(corpus, BM25_CONFIG)

    def retrieve_fn(text, k):
        return [sd.doc_id for sd in retriever.retrieve(text, top_k=k)]

    result = evaluate_retrieval(claims, retrieve_fn, k_values=[1, 3])
    assert result.num_claims_scored == 1
    assert result.num_claims_excluded_no_gold == 1
    assert result.recall_at_k[1] == 1.0  # doc 1 should be top hit for its own near-exact claim


def test_evaluate_retrieval_raises_if_all_claims_lack_gold(corpus):
    claims = [Claim(id=1, claim="No evidence.", evidence={}, cited_doc_ids=[])]
    retriever = BM25Retriever(corpus, BM25_CONFIG)

    def retrieve_fn(text, k):
        return [sd.doc_id for sd in retriever.retrieve(text, top_k=k)]

    with pytest.raises(ValueError):
        evaluate_retrieval(claims, retrieve_fn, k_values=[1])
