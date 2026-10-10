"""Pick the abstract sentences most relevant to a claim.

Why: the baseline fed the NLI model the whole title + abstract. Long abstracts
get truncated by the model's input limit and the relevant sentence is diluted
by unrelated ones. Giving the NLI model only the best-matching sentences is
the main change in the proposed system.

Method (deliberately simple and dependency-free): score each sentence by how
many distinct claim tokens it contains, break ties by earlier position, keep
the top-n and return them in their original order. The title is always kept.
"""

from __future__ import annotations

from typing import List, Tuple

from data.schema import CorpusDocument
from retrieval.tokenize import simple_lowercase_alnum


def select_sentences(claim: str, doc: CorpusDocument, n: int = 2) -> Tuple[str, List[int]]:
    """Return (evidence_text, selected_sentence_indices) for one document."""
    if n <= 0:
        raise ValueError("n must be positive")
    claim_tokens = set(simple_lowercase_alnum(claim))

    scored = []
    for idx, sentence in enumerate(doc.abstract):
        overlap = len(claim_tokens & set(simple_lowercase_alnum(sentence)))
        scored.append((-overlap, idx))          # more overlap first, then earlier sentence
    scored.sort()
    chosen = sorted(idx for _, idx in scored[:n])

    text = " ".join([doc.title] + [doc.abstract[i] for i in chosen])
    return text, chosen
