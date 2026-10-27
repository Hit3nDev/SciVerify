"""Confidence calibration for the judge: temperature scaling and reliability measurement.

A model's softmax probability is not automatically the chance that it is right. Temperature
scaling (Guo et al., 2017) divides the logits by one number T, fitted on labelled examples the
model was not trained on, so that "80% sure" means right about 80% of the time. The argmax
never changes (the order of the classes is kept), only how sure the model claims to be.

Reliability is measured the standard way: sort predictions into confidence bins and compare
each bin's average confidence with its actual accuracy. The gap, weighted by bin size, is the
expected calibration error (ECE). Perfectly calibrated means ECE 0.

Pure numpy; no model is needed, so all of it is unit-tested.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def negative_log_likelihood(logits: np.ndarray, labels: Sequence[int], temperature: float = 1.0) -> float:
    probs = softmax(logits, temperature)
    picked = probs[np.arange(len(labels)), np.asarray(labels)]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def fit_temperature(logits: np.ndarray, labels: Sequence[int]) -> float:
    """The T that minimises NLL on these examples, by a coarse then a fine grid search.
    One parameter, so a grid is enough and there is nothing to tune."""
    logits = np.asarray(logits, dtype=float)
    if len(logits) == 0:
        raise ValueError("no examples to fit on")
    coarse = np.arange(0.05, 10.0, 0.05)
    best = min(coarse, key=lambda t: negative_log_likelihood(logits, labels, t))
    fine = np.arange(max(0.01, best - 0.05), best + 0.05, 0.001)
    return float(min(fine, key=lambda t: negative_log_likelihood(logits, labels, t)))


def reliability_table(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> List[dict]:
    """Per confidence bin: how many predictions, their mean confidence, and their accuracy."""
    conf = np.asarray(confidences, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        rows.append({"lo": round(float(lo), 3), "hi": round(float(hi), 3), "count": n,
                     "confidence": round(float(conf[mask].mean()), 4) if n else None,
                     "accuracy": round(float(ok[mask].mean()), 4) if n else None})
    return rows


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    total = max(len(confidences), 1)
    return float(sum(r["count"] / total * abs(r["accuracy"] - r["confidence"])
                     for r in reliability_table(confidences, correct, n_bins) if r["count"]))


def summarize(logits: np.ndarray, labels: Sequence[int], temperature: float = 1.0, n_bins: int = 10) -> dict:
    """Everything the benchmarks page shows for one (model, temperature) pair."""
    probs = softmax(logits, temperature)
    pred = probs.argmax(axis=-1)
    conf = probs.max(axis=-1)
    ok = pred == np.asarray(labels)
    return {"temperature": round(float(temperature), 3), "n": int(len(labels)),
            "accuracy": round(float(ok.mean()), 4) if len(labels) else None,
            "nll": round(negative_log_likelihood(logits, labels, temperature), 4),
            "ece": round(expected_calibration_error(conf, ok, n_bins), 4),
            "mean_confidence": round(float(conf.mean()), 4) if len(labels) else None,
            "bins": reliability_table(conf, ok, n_bins)}
