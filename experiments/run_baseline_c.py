"""Run Baseline C: retrieval + NLI verification over SciFact dev claims.

Usage:
    python -m experiments.run_baseline_c

Requires network access to huggingface.co on first run to download the NLI
model (configs/verification.yaml > nli.model_name, default
facebook/bart-large-mnli, ~1.6GB). This is a larger model than the dense
retrieval baseline's -- expect a slower first run.

Writes real results to experiments/results/baseline_c.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from data.loader import load_claims, load_config, load_corpus
from data.schema import Verdict
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.verification_metrics import evaluate_verification
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.embeddings import get_embedder
from verification.aggregate import DocumentVerdict, aggregate
from verification.nli import get_nli_fn


def build_retriever(retriever_name: str, corpus, retrieval_config):
    if retriever_name == "bm25":
        return BM25Retriever(corpus, retrieval_config["bm25"]), "bm25"
    if retriever_name == "dense":
        embed_fn = get_embedder(retrieval_config["dense"]["model_name"])
        return DenseRetriever(corpus, retrieval_config["dense"], embed_fn), "dense"
    raise ValueError(f"Unknown retriever '{retriever_name}' in configs/verification.yaml")


def main() -> None:
    data_config = load_config("configs/data.yaml")
    with open("configs/retrieval.yaml", "r", encoding="utf-8") as f:
        retrieval_config = yaml.safe_load(f)
    with open("configs/verification.yaml", "r", encoding="utf-8") as f:
        verification_config = yaml.safe_load(f)

    raw_dir = Path(data_config["paths"]["raw_dir"])
    paths = data_config["paths"]
    baseline_c_config = verification_config["baseline_c"]
    nli_config = verification_config["nli"]
    agg_config = verification_config["aggregation"]

    print("Loading corpus and dev claims...")
    corpus = load_corpus(raw_dir, paths["corpus_file"])
    dev_claims = load_claims(raw_dir, paths["claims_dev_file"], has_labels=True)
    print(f"  corpus documents: {len(corpus)}")
    print(f"  dev claims: {len(dev_claims)}")

    retriever_name = baseline_c_config["retriever"]
    print(f"Building {retriever_name} retriever (per configs/verification.yaml)...")
    retriever, retriever_name = build_retriever(retriever_name, corpus, retrieval_config)

    print(f"Loading NLI model: {nli_config['model_name']} (downloads on first run)...")
    nli_fn = get_nli_fn(nli_config["model_name"], nli_config["label_map"])

    evidence_top_k = baseline_c_config["evidence_top_k"]
    min_conf = agg_config["min_confidence_for_verdict"]

    # Only claims with gold evidence are meaningfully scorable for retrieval
    # recall (same convention as evaluate_retrieval); but for verification
    # accuracy we score ALL dev claims, including NEI ones, since UNKNOWN is
    # a valid and expected verdict for them.
    print(f"Running full pipeline on {len(dev_claims)} dev claims...")
    t0 = time.perf_counter()
    gold_verdicts, predicted_verdicts, per_claim_confidences = [], [], []

    for claim in dev_claims:
        retrieved = retriever.retrieve(claim.claim, top_k=evidence_top_k)
        per_document = []
        for scored_doc in retrieved:
            doc = corpus[scored_doc.doc_id]
            evidence_text = doc.title + " " + " ".join(doc.abstract)
            prediction = nli_fn(claim.claim, evidence_text)
            per_document.append(DocumentVerdict(doc_id=scored_doc.doc_id, prediction=prediction))

        result = aggregate(per_document, min_confidence_for_verdict=min_conf)
        gold_verdicts.append(claim.gold_verdict())
        predicted_verdicts.append(result.verdict)
        per_claim_confidences.append(result.confidence)

    total_time_s = time.perf_counter() - t0
    per_claim_latency_ms = (total_time_s / len(dev_claims)) * 1000

    verification_result = evaluate_verification(gold_verdicts, predicted_verdicts)

    # Retrieval quality achieved by whichever retriever this run used, for context.
    def retrieve_fn(claim_text: str, k: int):
        return [sd.doc_id for sd in retriever.retrieve(claim_text, top_k=k)]

    retrieval_result = evaluate_retrieval(
        dev_claims, retrieve_fn, k_values=retrieval_config["evaluation"]["recall_at_k"]
    )

    output = {
        "baseline": "Dense/BM25 retrieval + NLI (Baseline C)",
        "retriever_used": retriever_name,
        "nli_model": nli_config["model_name"],
        "evidence_top_k": evidence_top_k,
        "min_confidence_for_verdict": min_conf,
        "corpus_size": len(corpus),
        "num_dev_claims_total": len(dev_claims),
        "retrieval": {
            "recall_at_k": retrieval_result.recall_at_k,
            "mrr": retrieval_result.mrr,
        },
        "verification": {
            "accuracy": verification_result.accuracy,
            "macro_f1": verification_result.macro_f1,
            "per_class_precision": verification_result.per_class_precision,
            "per_class_recall": verification_result.per_class_recall,
            "per_class_f1": verification_result.per_class_f1,
            "support": verification_result.support,
        },
        "total_pipeline_time_seconds": round(total_time_s, 3),
        "avg_latency_per_claim_ms": round(per_claim_latency_ms, 2),
    }

    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_c.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
