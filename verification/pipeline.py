"""End-to-end claim verification: claim in, cited verdict out.

This is the system the evaluation found to be best (Baseline C): BM25 retrieval,
the whole abstract fed to an NLI model, per-paper judgements combined by the
conflict-aware aggregation rule. Retriever and NLI model are injected so the
logic can be tested without downloading any model.

Honest limits, repeated in every response (`caveat`):
  * it only searches the corpus it was given (5,183 SciFact biomedical abstracts);
  * `confidence` is the NLI model's probability, NOT a calibrated probability of being right.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from data.schema import CorpusDocument, RawLabel, Verdict
from retrieval.tokenize import simple_lowercase_alnum
from verification.aggregate import DocumentVerdict, aggregate
from verification.evidence_selection import select_sentences
from verification.nli import NliPrediction

MAX_CLAIM_CHARS = 1000
MAX_TOP_K = 10
CAVEAT = ("Searches only the abstracts it was given (SciFact: 5,183 biomedical abstracts). "
          "Confidence is the NLI model's probability, not a calibrated chance of being right. "
          "On the SciFact dev set this pipeline gives the right verdict for about 59% of claims.")
# Words too common to be worth highlighting in the evidence (display only; retrieval is unchanged).
_STOPWORDS = frozenset(
    "a an and are as at be been by can does for from has have in into is it its may more "
    "of on or than that the their these this to was were which with not no less".split())


@dataclass
class EvidenceItem:
    doc_id: int
    title: str
    retrieval_score: float
    nli_label: str          # entailment / contradiction / neutral (raw model output)
    nli_score: float        # model probability of that label
    judgement: str          # SUPPORT / CONTRADICT / NEUTRAL
    passage: str            # the sentences of the abstract that best match the claim (display only)
    probabilities: Dict[str, float] = field(default_factory=dict)  # all NLI class probabilities
    sentences: List[str] = field(default_factory=list)              # the full abstract
    highlighted: List[int] = field(default_factory=list)            # indices of the passage sentences
    calibrated_score: Optional[float] = None                        # nli_score after temperature scaling


@dataclass
class VerifyResponse:
    claim: str
    verdict: str
    confidence: float
    explanation: str
    evidence: List[EvidenceItem] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    caveat: str = CAVEAT
    claim_terms: List[str] = field(default_factory=list)   # claim words worth highlighting
    settings: Dict[str, float] = field(default_factory=dict)
    confidence_calibrated: Optional[float] = None          # same rule as confidence, on calibrated scores
    judge: str = ""                                        # which NLI model produced this

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sources"] = [{"doc_id": e.doc_id, "title": e.title} for e in self.evidence]
        d["retrieval_scores"] = [e.retrieval_score for e in self.evidence]
        d["verification_scores"] = [e.nli_score for e in self.evidence]
        return d


def claim_terms(text: str) -> List[str]:
    terms: List[str] = []
    for tok in simple_lowercase_alnum(text):
        if tok not in _STOPWORDS and (len(tok) > 2 or tok.isdigit()) and tok not in terms:
            terms.append(tok)
    return terms


def calibrated_confidence(verdict: str, per_doc: List[DocumentVerdict], threshold: float) -> Optional[float]:
    """The aggregation rule's confidence, recomputed on calibrated scores: the mean calibrated
    score of the papers that counted for the verdict (min of the two sides for CONFLICTING).
    Which papers count is still decided by the raw score, exactly as in `aggregate`."""
    def side(label: RawLabel) -> Optional[float]:
        vals = [dv.prediction.calibrated_score for dv in per_doc
                if dv.prediction.mapped_label == label and dv.prediction.score >= threshold]
        if not vals or any(v is None for v in vals):
            return None
        return sum(vals) / len(vals)

    sup, con = side(RawLabel.SUPPORT), side(RawLabel.CONTRADICT)
    if verdict == Verdict.SUPPORTED.value:
        return sup
    if verdict == Verdict.CONTRADICTED.value:
        return con
    if verdict == Verdict.CONFLICTING.value and sup is not None and con is not None:
        return min(sup, con)
    return None


class ClaimVerifier:
    def __init__(self, corpus: Dict[int, CorpusDocument], retriever,
                 nli_fn: Callable[[str, str], NliPrediction],
                 evidence_top_k: int = 3, min_confidence: float = 0.5, display_sentences: int = 2,
                 judge_name: str = "") -> None:
        self.corpus, self.retriever, self.nli_fn = corpus, retriever, nli_fn
        self.top_k, self.min_confidence, self.display_sentences = evidence_top_k, min_confidence, display_sentences
        self.judge_name = judge_name
        # One model, many web requests: run NLI calls one at a time instead of fighting over the CPU.
        self._nli_lock = threading.Lock()

    def _judge(self, claim: str, evidence_text: str) -> NliPrediction:
        with self._nli_lock:
            return self.nli_fn(claim, evidence_text)

    def verify(self, claim: str, top_k: Optional[int] = None,
               min_confidence: Optional[float] = None) -> VerifyResponse:
        for kind, payload in self.verify_steps(claim, top_k, min_confidence):
            if kind == "done":
                return payload
        raise RuntimeError("verify_steps ended without a result")  # pragma: no cover

    def verify_steps(self, claim: str, top_k: Optional[int] = None,
                     min_confidence: Optional[float] = None) -> Iterator[Tuple[str, object]]:
        """Run the pipeline one stage at a time so a UI can show progress.

        Yields ("retrieved", [{doc_id, title, retrieval_score}]), then ("judged", EvidenceItem) per
        paper, then ("done", VerifyResponse). verify() is this generator run to the end.
        """
        text = (claim or "").strip()
        if not text:
            raise ValueError("The claim is empty.")
        k = min(max(int(self.top_k if top_k is None else top_k), 1), MAX_TOP_K)
        threshold = min(max(float(self.min_confidence if min_confidence is None else min_confidence), 0.0), 1.0)
        truncated = len(text) > MAX_CLAIM_CHARS
        text = text[:MAX_CLAIM_CHARS]
        terms = claim_terms(text)
        settings = {"top_k": k, "min_confidence": threshold}
        t0 = time.perf_counter()

        # A BM25 score of 0 or below means no meaningful word overlap: treat as "nothing found".
        found = [sd for sd in self.retriever.retrieve(text, top_k=k) if sd.score > 0]
        yield "retrieved", [{"doc_id": sd.doc_id, "title": self.corpus[sd.doc_id].title,
                             "retrieval_score": round(float(sd.score), 3)} for sd in found]
        if not found:
            yield "done", VerifyResponse(
                claim=text, verdict=Verdict.UNKNOWN.value, confidence=0.0,
                explanation=("No paper in the corpus shares meaningful vocabulary with this claim, "
                             "so there is no evidence to judge it."),
                elapsed_seconds=round(time.perf_counter() - t0, 2), claim_terms=terms, settings=settings,
                judge=self.judge_name)
            return

        per_doc: List[DocumentVerdict] = []
        items: List[EvidenceItem] = []
        for sd in found:
            doc = self.corpus[sd.doc_id]
            pred = self._judge(text, doc.title + " " + " ".join(doc.abstract))
            per_doc.append(DocumentVerdict(doc_id=sd.doc_id, prediction=pred))
            passage, chosen = select_sentences(text, doc, self.display_sentences)
            judgement = pred.mapped_label.value if pred.mapped_label else "NEUTRAL"
            item = EvidenceItem(doc_id=sd.doc_id, title=doc.title,
                                retrieval_score=round(float(sd.score), 3),
                                nli_label=pred.raw_nli_label, nli_score=round(pred.score, 3),
                                judgement=judgement, passage=passage,
                                probabilities={lbl: round(p, 4) for lbl, p in pred.probs.items()},
                                sentences=list(doc.abstract), highlighted=chosen,
                                calibrated_score=None if pred.calibrated_score is None else round(pred.calibrated_score, 3))
            items.append(item)
            yield "judged", item

        agg = aggregate(per_doc, min_confidence_for_verdict=threshold)
        explanation = agg.explanation
        if truncated:
            explanation += f" (The claim was longer than {MAX_CLAIM_CHARS} characters and was shortened.)"
        cal = calibrated_confidence(agg.verdict.value, per_doc, threshold)
        yield "done", VerifyResponse(claim=text, verdict=agg.verdict.value, confidence=round(agg.confidence, 3),
                                     explanation=explanation, evidence=items,
                                     elapsed_seconds=round(time.perf_counter() - t0, 2),
                                     claim_terms=terms, settings=settings,
                                     confidence_calibrated=None if cal is None else round(cal, 3),
                                     judge=self.judge_name)

    def explain_sentences(self, claim: str, doc_id: int) -> List[dict]:
        """Judge every sentence of one abstract on its own, to locate the sentences that decide it.

        Display aid only: the verdict itself is always computed from the whole abstract.
        """
        text = (claim or "").strip()[:MAX_CLAIM_CHARS]
        if not text:
            raise ValueError("The claim is empty.")
        if doc_id not in self.corpus:
            raise KeyError(doc_id)
        out = []
        for i, sentence in enumerate(self.corpus[doc_id].abstract):
            pred = self._judge(text, sentence)
            out.append({"index": i, "text": sentence, "nli_label": pred.raw_nli_label,
                        "score": round(pred.score, 4),
                        "judgement": pred.mapped_label.value if pred.mapped_label else "NEUTRAL",
                        "probabilities": {lbl: round(p, 4) for lbl, p in pred.probs.items()}})
        return out

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """Retrieval only (no model): the papers BM25 ranks highest for a query."""
        text = (query or "").strip()[:MAX_CLAIM_CHARS]
        if not text:
            return []
        hits = [sd for sd in self.retriever.retrieve(text, top_k=min(max(top_k, 1), 50)) if sd.score > 0]
        results = []
        for sd in hits:
            doc = self.corpus[sd.doc_id]
            chosen = select_sentences(text, doc, 1)[1] if doc.abstract else []
            results.append({"doc_id": sd.doc_id, "title": doc.title, "score": round(float(sd.score), 3),
                            "snippet": doc.abstract[chosen[0]] if chosen else ""})
        return results
