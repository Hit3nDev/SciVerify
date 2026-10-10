"""Hybrid retrieval: fuse several retrievers with Reciprocal Rank Fusion (RRF).

RRF scores a document as sum over retrievers of 1 / (rrf_k + rank). It uses
only ranks, so BM25 scores and cosine similarities (different scales) never
have to be normalised against each other.

Whether fusion actually helps on SciFact is an open question: BM25 alone
already reaches Recall@10 = 0.936 and dense retrieval scored lower, so any
gain here may be small. The experiment runner makes this switchable so the
ablation can report the honest answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class ScoredDoc:
    doc_id: int
    score: float


class HybridRetriever:
    def __init__(self, retrievers: Sequence, rrf_k: int = 60, top_k: int = 10,
                 candidate_k: int = 50) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self._retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.top_k = top_k
        self.candidate_k = candidate_k

    def retrieve(self, claim_text: str, top_k: int | None = None) -> List[ScoredDoc]:
        k = top_k if top_k is not None else self.top_k
        fused: Dict[int, float] = {}
        for retriever in self._retrievers:
            ranked = retriever.retrieve(claim_text, top_k=self.candidate_k)
            for rank, scored in enumerate(ranked, start=1):
                fused[scored.doc_id] = fused.get(scored.doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        results = [ScoredDoc(doc_id=d, score=s) for d, s in fused.items()]
        results.sort(key=lambda sd: (-sd.score, sd.doc_id))  # doc_id tie-break = deterministic
        return results[:k]
