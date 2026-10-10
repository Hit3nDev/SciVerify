"""Tests for retrieval/hybrid.py and verification/evidence_selection.py."""
import pytest

from data.schema import CorpusDocument
from retrieval.hybrid import HybridRetriever, ScoredDoc
from verification.evidence_selection import select_sentences


class FakeRetriever:
    def __init__(self, ranking):
        self.ranking = ranking

    def retrieve(self, text, top_k=10):
        return [ScoredDoc(d, 1.0) for d in self.ranking[:top_k]]


def test_rrf_prefers_document_ranked_well_by_both():
    a = FakeRetriever([1, 2, 3])
    b = FakeRetriever([2, 1, 4])
    out = HybridRetriever([a, b], rrf_k=60, top_k=4).retrieve("q")
    assert {out[0].doc_id, out[1].doc_id} == {1, 2}   # both lists agree on these two


def test_rrf_includes_docs_found_by_only_one_retriever():
    out = HybridRetriever([FakeRetriever([1]), FakeRetriever([9])], top_k=5).retrieve("q")
    assert {d.doc_id for d in out} == {1, 9}


def test_rrf_respects_top_k_and_is_deterministic():
    r = HybridRetriever([FakeRetriever([1, 2, 3, 4])], top_k=2)
    assert [d.doc_id for d in r.retrieve("q")] == [1, 2]
    assert [d.doc_id for d in r.retrieve("q")] == [1, 2]


def test_hybrid_requires_a_retriever():
    with pytest.raises(ValueError):
        HybridRetriever([])


def _doc():
    return CorpusDocument(doc_id=1, title="Exercise study", abstract=[
        "Participants were recruited from three clinics.",
        "Regular exercise improved memory scores in older adults.",
        "Funding was provided by a national grant.",
        "Exercise also reduced reported fatigue.",
    ])


def test_selection_picks_overlapping_sentences_in_original_order():
    text, idx = select_sentences("Exercise improves memory in older adults", _doc(), n=2)
    assert idx == [1, 3]
    assert text.startswith("Exercise study")
    assert "recruited" not in text and "grant" not in text


def test_selection_with_no_overlap_falls_back_to_first_sentences():
    _, idx = select_sentences("zzz qqq", _doc(), n=2)
    assert idx == [0, 1]


def test_selection_n_larger_than_abstract_keeps_everything():
    _, idx = select_sentences("exercise", _doc(), n=99)
    assert idx == [0, 1, 2, 3]


def test_selection_rejects_bad_n():
    with pytest.raises(ValueError):
        select_sentences("x", _doc(), n=0)
