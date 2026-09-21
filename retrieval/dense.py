"""Baseline B: dense/semantic retrieval over the SciFact corpus.

Embeds each corpus document (title + abstract) and each claim with the
same embedder, indexes document vectors in a FAISS flat inner-product
index (cosine similarity, since vectors are L2-normalized first), and
retrieves the top-K nearest documents per claim.

The embedder is injected (see retrieval/embeddings.py) rather than
constructed inside this class, so this module and its tests never need to
touch sentence-transformers or the network directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import faiss
import numpy as np

from data.schema import CorpusDocument
from retrieval.embeddings import EmbedFn


@dataclass
class ScoredDoc:
    doc_id: int
    score: float


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class DenseRetriever:
    """FAISS-backed dense retriever over a SciFact corpus."""

    def __init__(self, corpus: Dict[int, CorpusDocument], config: dict, embed_fn: EmbedFn) -> None:
        self._doc_ids: List[int] = list(corpus.keys())
        self._embed_fn = embed_fn
        self.top_k: int = config["top_k"]

        texts = [self._doc_text(corpus[doc_id]) for doc_id in self._doc_ids]
        embeddings = _l2_normalize(embed_fn(texts))
        dim = embeddings.shape[1]

        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    @staticmethod
    def _doc_text(doc: CorpusDocument) -> str:
        return doc.title + " " + " ".join(doc.abstract)

    def retrieve(self, claim_text: str, top_k: int | None = None) -> List[ScoredDoc]:
        k = top_k if top_k is not None else self.top_k
        k = min(k, len(self._doc_ids))
        query_vec = _l2_normalize(self._embed_fn([claim_text]))

        scores, indices = self._index.search(query_vec, k)
        results = [
            ScoredDoc(doc_id=self._doc_ids[idx], score=float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]
        return results
