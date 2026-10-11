"""Measure how well sentence selection finds the gold evidence sentences.

Needs only the corpus and dev claims (no models, runs in seconds). For every
dev claim with gold evidence and every gold document, compare the sentences
chosen by select_sentences() with the annotated rationale sentences, for
several values of n. A 'first n sentences' baseline is reported next to it so
the number has something to be compared with.

    python -m experiments.eval_sentence_selection

Purpose: the proposed system scored below Baseline C. If selection often misses
the gold sentences, that is a direct, testable explanation for the drop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from data.loader import load_claims, load_config, load_corpus
from data.schema import Claim, CorpusDocument
from verification.evidence_selection import select_sentences


def evaluate_selection(claims: List[Claim], corpus: Dict[int, CorpusDocument], n_values: List[int]) -> dict:
    results = {}
    for n in n_values:
        hits = recall_sum = first_hits = first_recall_sum = pairs = 0
        for claim in claims:
            for doc_id, rationales in claim.evidence.items():
                doc = corpus.get(int(doc_id))
                gold = {i for r in rationales for i in r.sentences}
                if doc is None or not gold:
                    continue
                pairs += 1
                _, chosen = select_sentences(claim.claim, doc, n)
                chosen_set, first_set = set(chosen), set(range(min(n, len(doc.abstract))))
                hits += bool(gold & chosen_set)
                recall_sum += len(gold & chosen_set) / len(gold)
                first_hits += bool(gold & first_set)
                first_recall_sum += len(gold & first_set) / len(gold)
        if pairs == 0:
            raise ValueError("no (claim, gold document) pairs found")
        results[str(n)] = {
            "pairs": pairs,
            "selected_hit_rate": hits / pairs,
            "selected_gold_sentence_recall": recall_sum / pairs,
            "first_n_hit_rate": first_hits / pairs,
            "first_n_gold_sentence_recall": first_recall_sum / pairs,
        }
    return results


def main() -> None:
    cfg = load_config("configs/data.yaml")
    raw = Path(cfg["paths"]["raw_dir"])
    corpus = load_corpus(raw, cfg["paths"]["corpus_file"])
    dev = load_claims(raw, cfg["paths"]["claims_dev_file"], has_labels=True)
    res = evaluate_selection(dev, corpus, [1, 2, 3, 4, 5])
    print(f"{'n':>2} {'pairs':>6} {'hit(select)':>12} {'hit(first n)':>13} {'gold-recall(select)':>20} {'gold-recall(first n)':>21}")
    for n, r in res.items():
        print(f"{n:>2} {r['pairs']:>6} {r['selected_hit_rate']:>12.3f} {r['first_n_hit_rate']:>13.3f} "
              f"{r['selected_gold_sentence_recall']:>20.3f} {r['first_n_gold_sentence_recall']:>21.3f}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/sentence_selection_eval.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
