"""Typed data structures for SciFact records.

Mirrors the schema documented at
https://github.com/allenai/scifact/blob/master/doc/data.md exactly, so that
downstream code never has to guess field names. Internal verdict labels
(SUPPORTED / CONTRADICTED / UNKNOWN / CONFLICTING) are kept separate from
the raw dataset labels (SUPPORT / CONTRADICT) — see configs/data.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class RawLabel(str, Enum):
    """Labels as they appear in the raw SciFact claim files."""

    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"


class Verdict(str, Enum):
    """Internal verdict vocabulary produced by SciVerify's verification stage."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"
    CONFLICTING = "CONFLICTING"  # see research spec section 12


@dataclass(frozen=True)
class Rationale:
    """A single evidence rationale: a set of sentence indices within one abstract
    plus the label that rationale supports."""

    sentences: List[int]
    label: RawLabel


@dataclass(frozen=True)
class Claim:
    """One SciFact claim record.

    `evidence` maps a cited document's S2ORC id (as a string, matching the raw
    JSON keys) to a list of Rationale objects. A claim with no evidence for any
    cited/retrieved document has an empty `evidence` dict (i.e. NEI/UNKNOWN).
    """

    id: int
    claim: str
    evidence: Dict[str, List[Rationale]]
    cited_doc_ids: List[int]

    def gold_doc_ids(self) -> List[int]:
        """Document ids that carry direct evidence (SUPPORT or CONTRADICT)."""
        return [int(doc_id) for doc_id in self.evidence.keys()]

    def has_conflicting_evidence(self) -> bool:
        """True if different cited documents give this claim opposite labels."""
        labels = set()
        for rationales in self.evidence.values():
            labels.update(r.label for r in rationales)
        return RawLabel.SUPPORT in labels and RawLabel.CONTRADICT in labels


@dataclass(frozen=True)
class CorpusDocument:
    """One abstract in the SciFact evidence corpus."""

    doc_id: int
    title: str
    abstract: List[str]  # list of sentences
    structured: bool = False


@dataclass
class RetrievedEvidence:
    """Output of the retrieval stage for one (claim, document) pair — populated
    at inference time, not part of the raw dataset."""

    doc_id: int
    title: str
    score: float
    retrieval_method: str  # "bm25" | "dense" | "hybrid"
    top_sentences: List[str] = field(default_factory=list)
    sentence_indices: List[int] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Final SciVerify output for one claim."""

    claim: str
    verdict: Verdict
    confidence: float
    evidence: List[RetrievedEvidence]
    explanation: str
    per_document_labels: Optional[Dict[int, str]] = None  # doc_id -> SUPPORT/CONTRADICT/NEI
