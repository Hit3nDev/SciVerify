"""Aggregate per-document NLI predictions into one claim-level verdict.

Implements research spec section 12 (conflicting evidence): a claim is not
simply "supported" because one retrieved abstract entails it. If different
retrieved abstracts disagree (one SUPPORT, one CONTRADICT), the system must
surface that explicitly as CONFLICTING rather than picking a side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from data.schema import RawLabel, Verdict
from verification.nli import NliPrediction


@dataclass
class DocumentVerdict:
    doc_id: int
    prediction: NliPrediction


@dataclass
class AggregatedResult:
    verdict: Verdict
    confidence: float
    per_document: List[DocumentVerdict]
    explanation: str


def aggregate(
    per_document: List[DocumentVerdict], min_confidence_for_verdict: float = 0.5
) -> AggregatedResult:
    """Combine per-document NLI predictions into a single verdict.

    Rules (documented, not implicit):
    1. No documents at all -> UNKNOWN, confidence 0.
    2. All documents agree (or only one has a non-neutral label) -> that
       label, confidence = mean score of the agreeing predictions.
    3. Documents disagree (some SUPPORT, some CONTRADICT) -> CONFLICTING,
       regardless of how many are on each side. Confidence reflects how
       balanced the conflict is (lower when evenly split).
    4. All documents are neutral / below min_confidence_for_verdict -> UNKNOWN.
    """
    if not per_document:
        return AggregatedResult(
            verdict=Verdict.UNKNOWN,
            confidence=0.0,
            per_document=[],
            explanation="No evidence documents were retrieved for this claim.",
        )

    support_docs = [
        dv for dv in per_document
        if dv.prediction.mapped_label == RawLabel.SUPPORT
        and dv.prediction.score >= min_confidence_for_verdict
    ]
    contradict_docs = [
        dv for dv in per_document
        if dv.prediction.mapped_label == RawLabel.CONTRADICT
        and dv.prediction.score >= min_confidence_for_verdict
    ]

    if support_docs and contradict_docs:
        # Genuine conflict — do not force a binary answer (spec section 12).
        support_conf = sum(dv.prediction.score for dv in support_docs) / len(support_docs)
        contradict_conf = sum(dv.prediction.score for dv in contradict_docs) / len(contradict_docs)
        # Confidence in "there is a real conflict" is high when both sides
        # are individually confident; low (near 0.5) when scores are weak.
        confidence = min(support_conf, contradict_conf)
        return AggregatedResult(
            verdict=Verdict.CONFLICTING,
            confidence=confidence,
            per_document=per_document,
            explanation=(
                f"{len(support_docs)} retrieved document(s) support the claim while "
                f"{len(contradict_docs)} contradict it. The evidence is genuinely mixed; "
                "this is reported as CONFLICTING rather than resolved to a single side."
            ),
        )

    if support_docs:
        confidence = sum(dv.prediction.score for dv in support_docs) / len(support_docs)
        return AggregatedResult(
            verdict=Verdict.SUPPORTED,
            confidence=confidence,
            per_document=per_document,
            explanation=(
                f"{len(support_docs)} retrieved document(s) entail the claim with no "
                "contradicting evidence found among retrieved documents."
            ),
        )

    if contradict_docs:
        confidence = sum(dv.prediction.score for dv in contradict_docs) / len(contradict_docs)
        return AggregatedResult(
            verdict=Verdict.CONTRADICTED,
            confidence=confidence,
            per_document=per_document,
            explanation=(
                f"{len(contradict_docs)} retrieved document(s) contradict the claim with no "
                "supporting evidence found among retrieved documents."
            ),
        )

    return AggregatedResult(
        verdict=Verdict.UNKNOWN,
        confidence=0.0,
        per_document=per_document,
        explanation=(
            "Retrieved documents did not provide confident entailment or contradiction "
            "evidence for this claim; treated as not enough evidence."
        ),
    )
