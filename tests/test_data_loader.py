"""Tests for data/loader.py and data/schema.py.

These use small synthetic fixtures that match SciFact's documented schema
byte-for-byte (see doc/data.md), because this development sandbox cannot
reach AllenAI's S3 bucket to fetch the real dataset. When run on a machine
with network access, point `raw_dir` at the real downloaded data instead —
these same functions are what Phase 1's real dataset_summary() call uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.loader import dataset_summary, load_claims, load_corpus
from data.schema import RawLabel

FIXTURE_CONFIG = {
    "paths": {
        "corpus_file": "corpus.jsonl",
        "claims_train_file": "claims_train.jsonl",
        "claims_dev_file": "claims_dev.jsonl",
        "claims_test_file": "claims_test.jsonl",
    }
}


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    corpus_rows = [
        {
            "doc_id": 11328820,
            "title": "Neutrophil extracellular traps and inflammation",
            "abstract": ["Sentence zero.", "Sentence one.", "Sentence two."],
            "structured": False,
        },
        {
            "doc_id": 30041340,
            "title": "Citrullination and disease",
            "abstract": ["Sentence zero.", "Sentence one."],
            "structured": False,
        },
    ]
    claims_train_rows = [
        {
            "id": 1,
            "claim": "Exercise improves cognitive performance in older adults.",
            "evidence": {
                "11328820": [{"sentences": [0, 1], "label": "SUPPORT"}],
            },
            "cited_doc_ids": [11328820],
        },
        {
            # Conflicting evidence: one doc supports, another contradicts.
            "id": 2,
            "claim": "Citrullinated proteins disrupt the inflammatory cycle.",
            "evidence": {
                "11328820": [{"sentences": [0], "label": "CONTRADICT"}],
                "30041340": [{"sentences": [1], "label": "SUPPORT"}],
            },
            "cited_doc_ids": [11328820, 30041340],
        },
        {
            # No evidence at all -> NEI / UNKNOWN.
            "id": 3,
            "claim": "An unrelated claim with no retrievable evidence.",
            "evidence": {},
            "cited_doc_ids": [],
        },
    ]
    claims_dev_rows = claims_train_rows[:2]
    claims_test_rows = [{"id": 4, "claim": "Unlabeled test claim.", "cited_doc_ids": []}]

    for fname, rows in [
        ("corpus.jsonl", corpus_rows),
        ("claims_train.jsonl", claims_train_rows),
        ("claims_dev.jsonl", claims_dev_rows),
        ("claims_test.jsonl", claims_test_rows),
    ]:
        with open(tmp_path / fname, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    return tmp_path


def test_load_corpus(raw_dir: Path):
    corpus = load_corpus(raw_dir, "corpus.jsonl")
    assert len(corpus) == 2
    assert corpus[11328820].title.startswith("Neutrophil")
    assert corpus[11328820].abstract == ["Sentence zero.", "Sentence one.", "Sentence two."]


def test_load_claims_with_labels(raw_dir: Path):
    claims = load_claims(raw_dir, "claims_train.jsonl", has_labels=True)
    assert len(claims) == 3
    claim1 = next(c for c in claims if c.id == 1)
    assert claim1.evidence["11328820"][0].label == RawLabel.SUPPORT


def test_load_claims_without_labels(raw_dir: Path):
    claims = load_claims(raw_dir, "claims_test.jsonl", has_labels=False)
    assert len(claims) == 1
    assert claims[0].evidence == {}


def test_conflicting_evidence_detection(raw_dir: Path):
    claims = load_claims(raw_dir, "claims_train.jsonl", has_labels=True)
    claim2 = next(c for c in claims if c.id == 2)
    assert claim2.has_conflicting_evidence() is True
    claim1 = next(c for c in claims if c.id == 1)
    assert claim1.has_conflicting_evidence() is False


def test_nei_claim_has_no_gold_docs(raw_dir: Path):
    claims = load_claims(raw_dir, "claims_train.jsonl", has_labels=True)
    claim3 = next(c for c in claims if c.id == 3)
    assert claim3.gold_doc_ids() == []


def test_dataset_summary_counts_are_real_not_hardcoded(raw_dir: Path):
    summary = dataset_summary(raw_dir, FIXTURE_CONFIG)
    assert summary["num_corpus_documents"] == 2
    assert summary["num_claims_train"] == 3
    assert summary["num_claims_dev"] == 2
    assert summary["num_claims_test"] == 1
    assert summary["train_label_counts"]["CONFLICTING"] == 1
    assert summary["train_label_counts"]["NEI"] == 1
    assert summary["train_label_counts"]["SUPPORT"] == 1


def test_missing_file_raises_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path, "corpus.jsonl")
