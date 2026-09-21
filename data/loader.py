"""Load and validate SciFact JSONL files into typed objects.

No metrics, statistics, or dataset sizes are hardcoded anywhere in this
module — everything is derived from the files on disk at load time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List

import yaml

from data.schema import Claim, CorpusDocument, RawLabel, Rationale


def load_config(config_path: str = "configs/data.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m data.download_data` first "
            "(requires network access to AllenAI's S3 bucket)."
        )
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_corpus(raw_dir: Path, corpus_filename: str) -> Dict[int, CorpusDocument]:
    """Returns {doc_id: CorpusDocument}."""
    docs: Dict[int, CorpusDocument] = {}
    for row in _iter_jsonl(raw_dir / corpus_filename):
        doc = CorpusDocument(
            doc_id=row["doc_id"],
            title=row["title"],
            abstract=row["abstract"],
            structured=row.get("structured", False),
        )
        docs[doc.doc_id] = doc
    return docs


def load_claims(raw_dir: Path, claims_filename: str, has_labels: bool = True) -> List[Claim]:
    """Load a claims_{train,dev,test}.jsonl file.

    The test file has no `evidence` field (unlabeled) — pass has_labels=False
    for it, and downstream code must not compute metrics against it.
    """
    claims: List[Claim] = []
    for row in _iter_jsonl(raw_dir / claims_filename):
        evidence: Dict[str, List[Rationale]] = {}
        if has_labels:
            for doc_id, rationale_list in row.get("evidence", {}).items():
                evidence[doc_id] = [
                    Rationale(sentences=r["sentences"], label=RawLabel(r["label"]))
                    for r in rationale_list
                ]
        claims.append(
            Claim(
                id=row["id"],
                claim=row["claim"],
                evidence=evidence,
                cited_doc_ids=row.get("cited_doc_ids", []),
            )
        )
    return claims


def dataset_summary(raw_dir: Path, config: dict) -> dict:
    """Compute real, on-disk statistics. Used by Phase 1 verification and by
    the research report — never populated with numbers from memory."""
    paths = config["paths"]
    corpus = load_corpus(raw_dir, paths["corpus_file"])
    train = load_claims(raw_dir, paths["claims_train_file"], has_labels=True)
    dev = load_claims(raw_dir, paths["claims_dev_file"], has_labels=True)
    test = load_claims(raw_dir, paths["claims_test_file"], has_labels=False)

    def label_counts(claims: List[Claim]) -> Dict[str, int]:
        counts = {"SUPPORT": 0, "CONTRADICT": 0, "NEI": 0, "CONFLICTING": 0}
        for c in claims:
            if not c.evidence:
                counts["NEI"] += 1
            elif c.has_conflicting_evidence():
                counts["CONFLICTING"] += 1
            else:
                labels = {r.label for rs in c.evidence.values() for r in rs}
                if RawLabel.SUPPORT in labels:
                    counts["SUPPORT"] += 1
                elif RawLabel.CONTRADICT in labels:
                    counts["CONTRADICT"] += 1
        return counts

    return {
        "num_corpus_documents": len(corpus),
        "num_claims_train": len(train),
        "num_claims_dev": len(dev),
        "num_claims_test": len(test),
        "train_label_counts": label_counts(train),
        "dev_label_counts": label_counts(dev),
    }
