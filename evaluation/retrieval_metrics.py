"""Retrieval evaluation metrics: Recall@K and Mean Reciprocal Rank (MRR).

A claim is scored against its gold evidence document ids
(Claim.gold_doc_ids()). Claims with zero gold docs (NEI claims — no
evidence exists to retrieve) are excluded from retrieval scoring by
default, since "recall" is undefined when there is nothing to recall —
this is a documented choice, not silently dropped data (see
`evaluate_retrieval`'s return value, which reports how many claims were
excluded and why).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from data.schema import Claim


@dataclass
class RetrievalEvalResult:
    recall_at_k: Dict[int, float]
    mrr: float
    num_claims_scored: int
    num_claims_excluded_no_gold: int


def recall_at_k(gold_ids: List[int], retrieved_ids: List[int], k: int) -> float:
    """1.0 if any gold doc is within the top-k retrieved, else 0.0."""
    if not gold_ids:
        raise ValueError("recall_at_k called with no gold ids — caller must filter these out")
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k.intersection(gold_ids) else 0.0


def reciprocal_rank(gold_ids: List[int], retrieved_ids: List[int]) -> float:
    """1/rank of the first retrieved doc that is in gold_ids, else 0.0."""
    if not gold_ids:
        raise ValueError("reciprocal_rank called with no gold ids — caller must filter these out")
    gold_set = set(gold_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    claims: List[Claim],
    retrieve_fn: Callable[[str, int], List[int]],
    k_values: List[int],
) -> RetrievalEvalResult:
    """Run a retriever over a list of claims and compute Recall@K + MRR.

    `retrieve_fn(claim_text, k)` must return a ranked list of doc_ids
    (highest-scored first), length up to `max(k_values)`.
    """
    max_k = max(k_values)
    recall_hits = {k: 0 for k in k_values}
    reciprocal_ranks: List[float] = []
    num_scored = 0
    num_excluded = 0

    for claim in claims:
        gold_ids = claim.gold_doc_ids()
        if not gold_ids:
            num_excluded += 1
            continue

        retrieved_ids = retrieve_fn(claim.claim, max_k)
        num_scored += 1

        for k in k_values:
            recall_hits[k] += recall_at_k(gold_ids, retrieved_ids, k)
        reciprocal_ranks.append(reciprocal_rank(gold_ids, retrieved_ids))

    if num_scored == 0:
        raise ValueError(
            "No claims had gold evidence documents to score against — "
            "check that the claims file was loaded with has_labels=True."
        )

    return RetrievalEvalResult(
        recall_at_k={k: recall_hits[k] / num_scored for k in k_values},
        mrr=sum(reciprocal_ranks) / num_scored,
        num_claims_scored=num_scored,
        num_claims_excluded_no_gold=num_excluded,
    )
