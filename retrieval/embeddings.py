"""Embedding model loading for dense retrieval.

The SentenceTransformer import and model download are deferred to inside
get_embedder() so that:
  1. Unit tests can inject a fake embed_fn without pulling in torch/HF.
  2. Importing retrieval.dense doesn't silently trigger a network call.

The model name is read from configs/retrieval.yaml (dense.model_name) —
never hardcoded elsewhere in the codebase.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np

EmbedFn = Callable[[List[str]], np.ndarray]


def get_embedder(model_name: str) -> EmbedFn:
    """Returns a function mapping a list of texts -> (N, dim) float32 array.

    Downloads `model_name` from the Hugging Face Hub on first use if not
    already cached locally. Requires network access to huggingface.co,
    which some sandboxed environments (including the one this project was
    drafted in) do not allow — see README > Known limitations.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. Run "
            "`pip install -r requirements.txt`."
        ) from exc

    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}'. If you're on a "
            "restricted network, this model needs to be downloaded from "
            "huggingface.co first. Original error: {exc}"
        ) from exc

    def embed(texts: List[str]) -> np.ndarray:
        return np.asarray(model.encode(texts, convert_to_numpy=True), dtype="float32")

    return embed
