"""Run the dense retrieval baseline (Baseline B) over SciFact dev claims.

Usage:
    python -m experiments.run_dense_baseline

Requires sentence-transformers to actually download configs/retrieval.yaml
> dense.model_name from Hugging Face on first run — needs normal internet
access (see retrieval/embeddings.py). Writes real results to
experiments/results/dense_baseline.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from data.loader import load_claims, load_config, load_corpus
from evaluation.retrieval_metrics import evaluate_retrieval
from retrieval.dense import DenseRetriever
from retrieval.embeddings import get_embedder


def main() -> None:
    data_config = load_config("configs/data.yaml")
    with open("configs/retrieval.yaml", "r", encoding="utf-8") as f:
        retrieval_config = yaml.safe_load(f)

    raw_dir = Path(data_config["paths"]["raw_dir"])
    paths = data_config["paths"]
    dense_config = retrieval_config["dense"]

    print("Loading corpus and dev claims...")
    corpus = load_corpus(raw_dir, paths["corpus_file"])
    dev_claims = load_claims(raw_dir, paths["claims_dev_file"], has_labels=True)
    print(f"  corpus documents: {len(corpus)}")
    print(f"  dev claims: {len(dev_claims)}")

    print(f"Loading embedder: {dense_config['model_name']} (downloads on first run)...")
    embed_fn = get_embedder(dense_config["model_name"])

    print("Building dense (FAISS) index...")
    t0 = time.perf_counter()
    retriever = DenseRetriever(corpus, dense_config, embed_fn)
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
        "baseline": "Dense retrieval (Baseline B)",
        "embedding_model": dense_config["model_name"],
        "corpus_size": len(corpus),
        "num_dev_claims_total": len(dev_claims),
        "num_claims_scored": result.num_claims_scored,
        "num_claims_excluded_no_gold_evidence": result.num_claims_excluded_no_gold,
        "recall_at_k": result.recall_at_k,
        "mrr": result.mrr,
        "index_build_time_seconds": round(index_time_s, 3),
        "total_eval_time_seconds": round(eval_time_s, 3),
        "avg_latency_per_claim_ms": round(per_claim_latency_ms, 2),
        "config": dense_config,
    }

    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dense_baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
