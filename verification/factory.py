"""Build the real verifier (loads the corpus and the NLI model). Needs the downloaded dataset
and the NLI model; everything is read from configs/, nothing is hardcoded here."""

from __future__ import annotations

from pathlib import Path

import yaml

from data.loader import load_claims, load_config, load_corpus
from retrieval.bm25 import BM25Retriever
from verification.nli import get_nli_fn
from verification.pipeline import ClaimVerifier


def build_verifier(nli_model: str | None = None) -> ClaimVerifier:
    data_cfg = load_config("configs/data.yaml")
    rcfg = yaml.safe_load(open("configs/retrieval.yaml", encoding="utf-8"))
    vcfg = yaml.safe_load(open("configs/verification.yaml", encoding="utf-8"))
    demo = vcfg["demo"]
    corpus = load_corpus(Path(data_cfg["paths"]["raw_dir"]), data_cfg["paths"]["corpus_file"])
    nli_fn = get_nli_fn(nli_model or demo["nli_model"], vcfg["nli"]["label_map"])
    return ClaimVerifier(corpus, BM25Retriever(corpus, rcfg["bm25"]), nli_fn,
                         evidence_top_k=demo["evidence_top_k"],
                         min_confidence=vcfg["aggregation"]["min_confidence_for_verdict"])


def load_labelled_claims(split: str = "dev") -> list[dict]:
    """The labelled SciFact claims of one split, as plain dicts for the web UI's
    "random claim" and batch features (the answer key is shown next to the system's verdict)."""
    paths = load_config("configs/data.yaml")["paths"]
    claims = load_claims(Path(paths["raw_dir"]), paths[f"claims_{split}_file"], has_labels=True)
    return [{"id": c.id, "claim": c.claim, "gold_verdict": c.gold_verdict().value,
             "gold_doc_ids": c.gold_doc_ids()} for c in claims]


def describe() -> dict:
    """What the demo runs, for the UI's status line."""
    vcfg = yaml.safe_load(open("configs/verification.yaml", encoding="utf-8"))
    return {"retriever": "BM25", "nli_model": vcfg["demo"]["nli_model"],
            "evidence_top_k": vcfg["demo"]["evidence_top_k"],
            "min_confidence": vcfg["aggregation"]["min_confidence_for_verdict"]}
