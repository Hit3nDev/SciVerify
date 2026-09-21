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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from data.schema import RawLabel

NliFn = Callable[[str, str], "NliPrediction"]


@dataclass
class NliPrediction:
    raw_nli_label: str                 # "entailment" | "contradiction" | "neutral"
    mapped_label: Optional[RawLabel]   # None means neutral/NEI (no RawLabel)
    score: float                       # softmax probability of raw_nli_label


def get_nli_fn(model_name: str, label_map: Dict[str, str]) -> NliFn:
    """Returns (claim, evidence_text) -> NliPrediction, backed by a real HF
    NLI model. Requires network access to huggingface.co on first call
    (model download) -- see README > Known limitations for why this can't be
    exercised in this development sandbox.
    """
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

    def predict(claim: str, evidence_text: str) -> NliPrediction:
        # Standard NLI framing: evidence is the premise, claim is the hypothesis.
        inputs = tokenizer(evidence_text, claim, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        top_idx = int(torch.argmax(probs).item())
        raw_label = id2label[top_idx]
        score = float(probs[top_idx].item())

        mapped = label_map.get(raw_label)  # e.g. "entailment" -> "SUPPORT"
        mapped_label = RawLabel(mapped) if mapped in ("SUPPORT", "CONTRADICT") else None

        return NliPrediction(raw_nli_label=raw_label, mapped_label=mapped_label, score=score)

    return predict
