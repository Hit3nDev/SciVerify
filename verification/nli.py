"""NLI-based claim verification.

Given a claim and an evidence passage, predicts whether the evidence
entails, contradicts, or is neutral toward the claim, then maps that to
SciVerify's internal RawLabel vocabulary via configs/verification.yaml.

Uses a standard MNLI-style sequence classification model directly
(AutoModelForSequenceClassification with premise=evidence, hypothesis=claim),
not a zero-shot pipeline -- that's the correct framing for a model like
facebook/bart-large-mnli, whose id2label is exactly
{0: contradiction, 1: neutral, 2: entailment}.

The model is loaded lazily (only inside get_nli_fn), same rationale as
retrieval/embeddings.py: keeps this module importable/testable without
network access, and callers inject a fn rather than a hardcoded model.

Two layers: get_nli_logits_fn returns the raw logits (what the calibration
script fits on), and get_nli_fn turns them into a labelled prediction. With a
fitted temperature (see verification/calibration.py) every prediction also
carries a calibrated score; the raw score is kept because every reported
accuracy number was measured with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from data.schema import RawLabel
from verification.calibration import softmax

NliFn = Callable[[str, str], "NliPrediction"]


@dataclass
class NliPrediction:
    raw_nli_label: str                 # "entailment" | "contradiction" | "neutral"
    mapped_label: Optional[RawLabel]   # None means neutral/NEI (no RawLabel)
    score: float                       # softmax probability of raw_nli_label
    probs: Dict[str, float] = field(default_factory=dict)  # all three class probabilities (UI only)
    calibrated_score: Optional[float] = None               # score after temperature scaling, if fitted


def get_nli_logits_fn(model_name: str) -> Tuple[Callable[[str, str], np.ndarray], Dict[int, str]]:
    """Returns ((claim, evidence_text) -> logits ndarray, id2label), backed by a real HF
    NLI model. `model_name` is a hub id or a local directory (e.g. models/nli-scifact).
    Requires network access to huggingface.co on first call for hub models."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers/torch are not installed. Run `pip install -r requirements.txt`."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load NLI model '{model_name}'. Requires network access "
            f"to huggingface.co. Original error: {exc}"
        ) from exc

    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}

    def logits_fn(claim: str, evidence_text: str) -> np.ndarray:
        # Standard NLI framing: evidence is the premise, claim is the hypothesis.
        inputs = tokenizer(evidence_text, claim, return_tensors="pt", truncation=True)
        with torch.no_grad():
            return model(**inputs).logits[0].detach().cpu().numpy().astype(float)

    return logits_fn, id2label


def prediction_from_logits(logits: np.ndarray, id2label: Dict[int, str], label_map: Dict[str, str],
                           temperature: Optional[float] = None) -> NliPrediction:
    """Pure function (tested without a model): logits -> labelled prediction."""
    probs = softmax(logits)
    top_idx = int(np.argmax(probs))
    raw_label = id2label[top_idx]
    mapped = label_map.get(raw_label)  # e.g. "entailment" -> "SUPPORT"
    mapped_label = RawLabel(mapped) if mapped in ("SUPPORT", "CONTRADICT") else None
    calibrated = float(softmax(logits, temperature)[top_idx]) if temperature else None
    return NliPrediction(raw_nli_label=raw_label, mapped_label=mapped_label, score=float(probs[top_idx]),
                         probs={id2label[i]: float(p) for i, p in enumerate(probs)},
                         calibrated_score=calibrated)


def get_nli_fn(model_name: str, label_map: Dict[str, str], temperature: Optional[float] = None) -> NliFn:
    """Returns (claim, evidence_text) -> NliPrediction, backed by a real HF NLI model."""
    logits_fn, id2label = get_nli_logits_fn(model_name)

    def predict(claim: str, evidence_text: str) -> NliPrediction:
        return prediction_from_logits(logits_fn(claim, evidence_text), id2label, label_map, temperature)

    return predict
