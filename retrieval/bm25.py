"""Baseline A: BM25 sparse/lexical retrieval over the SciFact corpus.

Indexes each corpus document as title + abstract sentences concatenated,
tokenized with the shared simple tokenizer (retrieval/tokenize.py). Given a
claim, returns the top-K scored documents.

This is intentionally the simplest possible retriever in the project —
see retrieval/tokenize.py for why — so that later baselines (dense,
hybrid) can be measured against a clean, well-understood floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from rank_bm25 import BM25Okapi

from data.schema import CorpusDocument
from retrieval.tokenize import get_tokenizer


@dataclass
class ScoredDoc:
    doc_id: int
    score: float


class BM25Retriever:
    """Wraps rank_bm25.BM25Okapi over a SciFact corpus."""

    def __init__(self, corpus: Dict[int, CorpusDocument], config: dict) -> None:
        self._doc_ids: List[int] = list(corpus.keys())
        self._tokenizer = get_tokenizer(config["tokenizer"])
        self.top_k: int = config["top_k"]

        tokenized_docs = [
            self._tokenizer(self._doc_text(corpus[doc_id])) for doc_id in self._doc_ids
        ]
        # rank_bm25's k1/b are set on the class instance after construction
        # in older versions; guard for the constructor kwarg too.
        try:
            self._bm25 = BM25Okapi(
                tokenized_docs, k1=config.get("k1", 1.5), b=config.get("b", 0.75)
            )
        except TypeError:
            self._bm25 = BM25Okapi(tokenized_docs)
            self._bm25.k1 = config.get("k1", 1.5)
            self._bm25.b = config.get("b", 0.75)

    @staticmethod
    def _doc_text(doc: CorpusDocument) -> str:
        return doc.title + " " + " ".join(doc.abstract)

    def retrieve(self, claim_text: str, top_k: int | None = None) -> List[ScoredDoc]:
        """Return the top_k documents for a claim, highest score first."""
        k = top_k if top_k is not None else self.top_k
        query_tokens = self._tokenizer(claim_text)
        scores = self._bm25.get_scores(query_tokens)

        scored = [
            ScoredDoc(doc_id=doc_id, score=float(score))
            for doc_id, score in zip(self._doc_ids, scores)
        ]
        scored.sort(key=lambda sd: sd.score, reverse=True)
        return scored[:k]
