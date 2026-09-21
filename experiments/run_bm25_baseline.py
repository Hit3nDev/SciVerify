"""Run the BM25 baseline (Baseline A) over SciFact dev claims and report
Recall@K / MRR.

Usage:
    python -m experiments.run_bm25_baseline

Writes results to experiments/results/bm25_baseline.json — real numbers
only, produced by actually running the retriever, never hand-written.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from data.loader import load_claims, load_config, load_corpus
from evaluation.retrieval_metrics import evaluate_retrieval
from retrieval.bm25 import BM25Retriever


def main() -> None:
    data_config = load_config("configs/data.yaml")
    with open("configs/retrieval.yaml", "r", encoding="utf-8") as f:
        retrieval_config = yaml.safe_load(f)

    raw_dir = Path(data_config["paths"]["raw_dir"])
    paths = data_config["paths"]

    print("Loading corpus and dev claims...")
    corpus = load_corpus(raw_dir, paths["corpus_file"])
    dev_claims = load_claims(raw_dir, paths["claims_dev_file"], has_labels=True)
    print(f"  corpus documents: {len(corpus)}")
    print(f"  dev claims: {len(dev_claims)}")

    print("Building BM25 index...")
    t0 = time.perf_counter()
    retriever = BM25Retriever(corpus, retrieval_config["bm25"])
    index_time_s = time.perf_counter() - t0
    print(f"  index build time: {index_time_s:.2f}s")

    def retrieve_fn(claim_text: str, k: int):
        return [sd.doc_id for sd in retriever.retrieve(claim_text, top_k=k)]

    print("Evaluating on dev set...")
    t0 = time.perf_counter()
    result = evaluate_retrieval(
        dev_claims, retrieve_fn, k_values=retrieval_config["evaluation"]["recall_at_k"]
    )
    eval_time_s = time.perf_counter() - t0
    per_claim_latency_ms = (eval_time_s / max(result.num_claims_scored, 1)) * 1000

    output = {
        "baseline": "BM25 (Baseline A)",
        "corpus_size": len(corpus),
        "num_dev_claims_total": len(dev_claims),
        "num_claims_scored": result.num_claims_scored,
        "num_claims_excluded_no_gold_evidence": result.num_claims_excluded_no_gold,
        "recall_at_k": result.recall_at_k,
        "mrr": result.mrr,
        "index_build_time_seconds": round(index_time_s, 3),
        "total_eval_time_seconds": round(eval_time_s, 3),
        "avg_latency_per_claim_ms": round(per_claim_latency_ms, 2),
        "config": retrieval_config["bm25"],
    }

    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bm25_baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
