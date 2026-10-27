"""Build the real verifier (loads the corpus and the NLI model). Needs the downloaded dataset
and the NLI model; everything is read from configs/, nothing is hardcoded here.

`build_verifier` gives one verifier (used by the CLI demo and the Streamlit page).
`VerifierPool` gives one verifier per judge in configs/verification.yaml > demo.judges, all
sharing a single corpus and BM25 index (used by the web app's judge switcher)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from data.loader import load_claims, load_config, load_corpus
from retrieval.bm25 import BM25Retriever
from verification.nli import get_nli_fn
from verification.pipeline import ClaimVerifier


def _vcfg() -> dict:
    return yaml.safe_load(open("configs/verification.yaml", encoding="utf-8"))


def load_temperatures(path: Optional[str] = None) -> Dict[str, float]:
    """{model path: fitted temperature} written by experiments/calibrate.py; empty if never run."""
    file = Path(path or _vcfg()["demo"].get("calibration_file", "models/calibration.json"))
    if not file.exists():
        return {}
    data = json.loads(file.read_text(encoding="utf-8"))
    return {k: float(v["temperature"]) for k, v in data.items() if "temperature" in v}


def _is_local(path: str) -> bool:
    return path.startswith(("models/", "./", "../")) or ":" in path or Path(path).exists()


def list_judges() -> List[dict]:
    """The judges the web app can offer: every configured one whose weights exist (local
    directories that were never trained are left out). Each has key, path, name, description."""
    demo = _vcfg()["demo"]
    judges = demo.get("judges") or [{"key": "default", "path": demo["nli_model"],
                                    "name": demo["nli_model"], "description": ""}]
    temps = load_temperatures(demo.get("calibration_file"))
    out = []
    for j in judges:
        if _is_local(j["path"]) and not Path(j["path"]).is_dir():
            continue
        out.append({**j, "calibrated": j["path"] in temps, "temperature": temps.get(j["path"])})
    return out


def _load_corpus_and_retriever():
    data_cfg = load_config("configs/data.yaml")
    rcfg = yaml.safe_load(open("configs/retrieval.yaml", encoding="utf-8"))
    corpus = load_corpus(Path(data_cfg["paths"]["raw_dir"]), data_cfg["paths"]["corpus_file"])
    return corpus, BM25Retriever(corpus, rcfg["bm25"])


def _make_verifier(model_path: str, corpus, retriever, name: str = "") -> ClaimVerifier:
    vcfg = _vcfg()
    demo = vcfg["demo"]
    temperature = load_temperatures(demo.get("calibration_file")).get(model_path)
    nli_fn = get_nli_fn(model_path, vcfg["nli"]["label_map"], temperature=temperature)
    return ClaimVerifier(corpus, retriever, nli_fn, evidence_top_k=demo["evidence_top_k"],
                         min_confidence=vcfg["aggregation"]["min_confidence_for_verdict"],
                         judge_name=name or model_path)


def build_verifier(nli_model: str | None = None) -> ClaimVerifier:
    corpus, retriever = _load_corpus_and_retriever()
    model_path = nli_model or _vcfg()["demo"]["nli_model"]
    name = next((j["name"] for j in list_judges() if j["path"] == model_path), model_path)
    return _make_verifier(model_path, corpus, retriever, name)


class VerifierPool:
    """One ClaimVerifier per judge key, built on demand, sharing the corpus and the index."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._shared = None
        self._verifiers: Dict[str, ClaimVerifier] = {}
        self.judges = list_judges()
        if not self.judges:
            raise RuntimeError("No judge is available; check configs/verification.yaml > demo.judges")
        self.default_key = self.judges[0]["key"]

    def judge(self, key: str) -> dict:
        for j in self.judges:
            if j["key"] == key:
                return j
        raise KeyError(key)

    def get(self, key: Optional[str] = None) -> ClaimVerifier:
        key = key or self.default_key
        j = self.judge(key)
        with self._lock:
            if self._shared is None:
                self._shared = _load_corpus_and_retriever()
            if key not in self._verifiers:
                self._verifiers[key] = _make_verifier(j["path"], *self._shared, name=j["name"])
            return self._verifiers[key]


def load_labelled_claims(split: str = "dev") -> list[dict]:
    """The labelled SciFact claims of one split, as plain dicts for the web UI's
    "random claim" and batch features (the answer key is shown next to the system's verdict)."""
    paths = load_config("configs/data.yaml")["paths"]
    claims = load_claims(Path(paths["raw_dir"]), paths[f"claims_{split}_file"], has_labels=True)
    return [{"id": c.id, "claim": c.claim, "gold_verdict": c.gold_verdict().value,
             "gold_doc_ids": c.gold_doc_ids()} for c in claims]


def describe() -> dict:
    """What the demo runs, for the UI's status line."""
    vcfg = _vcfg()
    return {"retriever": "BM25", "nli_model": vcfg["demo"]["nli_model"],
            "evidence_top_k": vcfg["demo"]["evidence_top_k"],
            "min_confidence": vcfg["aggregation"]["min_confidence_for_verdict"]}
