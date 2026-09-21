"""Minimal, dependency-free tokenizer shared by sparse retrieval.

Kept deliberately simple for Phase 2 (BM25 baseline) — no stemming, no
stopword removal, no domain-specific handling. This is a documented design
choice, not an oversight: a fancier tokenizer would blur the comparison
between the "traditional lexical retrieval" baseline (spec section 10) and
the dense/hybrid methods added in later phases. If ablations later show
BM25 underperforming due to vocabulary mismatch, upgrading this tokenizer
is the first thing to try — noted in experiments/ROADMAP.md.
"""

from __future__ import annotations

import re
from typing import List

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def simple_lowercase_alnum(text: str) -> List[str]:
    """Lowercase and split on runs of alphanumeric characters."""
    return _TOKEN_RE.findall(text.lower())


TOKENIZERS = {
    "simple_lowercase_alnum": simple_lowercase_alnum,
}


def get_tokenizer(name: str):
    if name not in TOKENIZERS:
        raise ValueError(f"Unknown tokenizer '{name}'. Available: {list(TOKENIZERS)}")
    return TOKENIZERS[name]
