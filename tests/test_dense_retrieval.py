"""Tests for retrieval/dense.py using a deterministic fake embedder.

We deliberately avoid loading a real sentence-transformers model here —
that requires network access this sandbox doesn't have (see
retrieval/embeddings.py docstring) and would make tests slow/non-hermetic
anyway. Instead we hand-craft embeddings where similarity is known in
advance, so we can assert exact retrieval behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.schema import CorpusDocument
from retrieval.dense import DenseRetriever

CONFIG = {"top_k": 2}


@pytest.fixture()
def corpus():
    return {
        1: CorpusDocument(doc_id=1, title="Doc A", abstract=["about exercise and cognition"]),
        2: CorpusDocument(doc_id=2, title="Doc B", abstract=["about sleep in teenagers"]),
        3: CorpusDocument(doc_id=3, title="Doc C", abstract=["about diet and heart disease"]),
    }


def _fake_embed_fn(text_to_vector: dict):
    """Builds an embed_fn that looks up precomputed vectors by exact text match."""

    def embed(texts):
        return np.array([text_to_vector[t] for t in texts], dtype="float32")

    return embed


def test_dense_retriever_ranks_by_cosine_similarity(corpus):
    # Doc A's text is closest to the query vector; Doc C is orthogonal/opposite.
    vectors = {
        "Doc A about exercise and cognition": [1.0, 0.0, 0.0],
        "Doc B about sleep in teenagers": [0.0, 1.0, 0.0],
        "Doc C about diet and heart disease": [-1.0, 0.0, 0.0],
        "does exercise help cognition?": [0.9, 0.1, 0.0],
    }
    embed_fn = _fake_embed_fn(vectors)
    retriever = DenseRetriever(corpus, CONFIG, embed_fn)

    results = retriever.retrieve("does exercise help cognition?", top_k=3)
    assert results[0].doc_id == 1
    assert results[-1].doc_id == 3  # most dissimilar (opposite vector) ranks last


def test_dense_retriever_respects_top_k(corpus):
    vectors = {
        "Doc A about exercise and cognition": [1.0, 0.0, 0.0],
        "Doc B about sleep in teenagers": [0.9, 0.1, 0.0],
        "Doc C about diet and heart disease": [0.8, 0.2, 0.0],
        "query": [1.0, 0.0, 0.0],
    }
    embed_fn = _fake_embed_fn(vectors)
    retriever = DenseRetriever(corpus, CONFIG, embed_fn)
    results = retriever.retrieve("query", top_k=2)
    assert len(results) == 2


def test_dense_retriever_top_k_capped_at_corpus_size(corpus):
    vectors = {
        "Doc A about exercise and cognition": [1.0, 0.0, 0.0],
        "Doc B about sleep in teenagers": [0.5, 0.5, 0.0],
        "Doc C about diet and heart disease": [0.0, 0.0, 1.0],
        "query": [1.0, 0.0, 0.0],
    }
    embed_fn = _fake_embed_fn(vectors)
    retriever = DenseRetriever(corpus, CONFIG, embed_fn)
    results = retriever.retrieve("query", top_k=100)  # larger than corpus
    assert len(results) == 3
