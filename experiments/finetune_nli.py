"""Fine-tune a small NLI judge on SciFact's labelled training claims.

Why: the error analysis (experiments/error_analysis.py) found that the weak part of the
pipeline is the judge, an NLI model that had never seen a scientific claim, not the
search. This teaches a small judge with the training split's 809 claims and their
annotated evidence. Only the TRAIN split is used, both for training and for choosing
the best epoch (a 10% slice of it is held out); the dev split is touched only by the
evaluation scripts afterwards, so the reported dev numbers stay honest.

    python -m experiments.finetune_nli                          # 30-40 min on a 12-core CPU
    python -m experiments.finetune_nli --limit 40 --epochs 1    # smoke test

Training examples, one per (claim, paper) pair, in the same form the judge sees at run time:
  * entailment / contradiction: a claim and a paper that the annotators marked as evidence.
    Each pair is added twice, once with the whole abstract and once with just the annotated
    rationale sentences, so the model learns both to find the key sentence and to read it.
  * neutral: a claim and a paper that carries no evidence for it; either a cited paper the
    annotators found nothing in, or a paper BM25 ranks highly that is not evidence. The
    second kind is exactly the mistake the pipeline makes most often (a verdict where the
    answer is "no evidence"), so the judge sees those hard cases during training.

Writes models/nli-scifact/ (loadable with verification.nli.get_nli_fn, same label names as
the base model) and experiments/results/finetune_nli.json (the training curve).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import yaml

from data.loader import load_claims, load_config, load_corpus
from data.schema import Claim, CorpusDocument, RawLabel
from retrieval.bm25 import BM25Retriever

RAW_TO_NLI = {RawLabel.SUPPORT: "entailment", RawLabel.CONTRADICT: "contradiction"}


def build_examples(claims: List[Claim], corpus: Dict[int, CorpusDocument], retriever,
                   negatives_per_claim: int = 2, retrieve_top_k: int = 3) -> List[dict]:
    """One training example per (claim, paper) pair. Pure function, tested without a model."""
    examples: List[dict] = []
    for claim in claims:
        for doc_id_str, rationales in claim.evidence.items():
            labels = {r.label for r in rationales}
            if len(labels) != 1:                      # a paper that both supports and contradicts: skip
                continue
            doc = corpus[int(doc_id_str)]
            nli_label = RAW_TO_NLI[labels.pop()]
            examples.append({"claim_id": claim.id, "doc_id": doc.doc_id, "claim": claim.claim,
                             "text": doc.title + " " + " ".join(doc.abstract),
                             "label": nli_label, "kind": "abstract"})
            idx = sorted({i for r in rationales for i in r.sentences if 0 <= i < len(doc.abstract)})
            if idx:
                examples.append({"claim_id": claim.id, "doc_id": doc.doc_id, "claim": claim.claim,
                                 "text": doc.title + " " + " ".join(doc.abstract[i] for i in idx),
                                 "label": nli_label, "kind": "rationale"})

        gold = set(claim.evidence.keys())
        neutral_ids: List[int] = [d for d in claim.cited_doc_ids if str(d) not in gold]
        for sd in retriever.retrieve(claim.claim, top_k=retrieve_top_k):
            if str(sd.doc_id) not in gold and sd.doc_id not in neutral_ids:
                neutral_ids.append(sd.doc_id)
        for doc_id in neutral_ids[:negatives_per_claim]:
            doc = corpus[doc_id]
            examples.append({"claim_id": claim.id, "doc_id": doc_id, "claim": claim.claim,
                             "text": doc.title + " " + " ".join(doc.abstract),
                             "label": "neutral", "kind": "neutral"})
    return examples


def split_by_claim(examples: List[dict], val_fraction: float, seed: int):
    """Hold out whole claims (never split one claim's pairs across train and validation)."""
    ids = sorted({e["claim_id"] for e in examples})
    random.Random(seed).shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction)) if len(ids) > 1 else 0
    val_ids = set(ids[:n_val])
    return [e for e in examples if e["claim_id"] not in val_ids], [e for e in examples if e["claim_id"] in val_ids]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="cross-encoder/nli-MiniLM2-L6-H768")
    ap.add_argument("--out", default="models/nli-scifact")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=320)
    ap.add_argument("--negatives-per-claim", type=int, default=2)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--limit", type=int, help="only the first N training claims (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import numpy as np
    import torch
    from sklearn.metrics import f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)

    data_cfg = load_config("configs/data.yaml")
    rcfg = yaml.safe_load(open("configs/retrieval.yaml", encoding="utf-8"))
    raw_dir = Path(data_cfg["paths"]["raw_dir"])
    corpus = load_corpus(raw_dir, data_cfg["paths"]["corpus_file"])
    train_claims = load_claims(raw_dir, data_cfg["paths"]["claims_train_file"], has_labels=True)
    if args.limit:
        train_claims = train_claims[: args.limit]

    print(f"Building examples from {len(train_claims)} training claims...")
    examples = build_examples(train_claims, corpus, BM25Retriever(corpus, rcfg["bm25"]),
                              negatives_per_claim=args.negatives_per_claim)
    train_ex, val_ex = split_by_claim(examples, args.val_fraction, args.seed)
    counts = {"train": dict(Counter(e["label"] for e in train_ex)), "val": dict(Counter(e["label"] for e in val_ex)),
              "by_kind": dict(Counter(e["kind"] for e in examples))}
    print(f"  {len(train_ex)} train / {len(val_ex)} validation examples: {counts}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model)
    label2id = {k.lower(): int(v) for k, v in model.config.label2id.items()}
    for lbl in ("entailment", "contradiction", "neutral"):
        if lbl not in label2id:
            raise SystemExit(f"base model must have an '{lbl}' label; it has {list(label2id)}")

    # Rarer labels count more, so the model is not rewarded for always saying "entailment".
    freq = Counter(e["label"] for e in train_ex)
    weights = torch.tensor([1.0 / (freq[lbl] ** 0.5) for lbl, _ in sorted(label2id.items(), key=lambda kv: kv[1])])
    weights = weights / weights.mean()
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def batches(rows: List[dict], shuffle: bool):
        order = list(range(len(rows)))
        if shuffle:
            random.shuffle(order)
        for i in range(0, len(order), args.batch_size):
            chunk = [rows[j] for j in order[i:i + args.batch_size]]
            enc = tokenizer([c["text"] for c in chunk], [c["claim"] for c in chunk], return_tensors="pt",
                            truncation="only_first", max_length=args.max_length, padding=True)
            yield enc, torch.tensor([label2id[c["label"]] for c in chunk])

    def evaluate(rows: List[dict]) -> dict:
        model.eval()
        preds, golds, total_loss = [], [], 0.0
        with torch.no_grad():
            for enc, y in batches(rows, shuffle=False):
                logits = model(**enc).logits
                total_loss += float(loss_fn(logits, y)) * len(y)
                preds += logits.argmax(-1).tolist()
                golds += y.tolist()
        model.train()
        return {"loss": round(total_loss / max(len(rows), 1), 4),
                "accuracy": round(float(np.mean(np.array(preds) == np.array(golds))), 4) if rows else None,
                "macro_f1": round(float(f1_score(golds, preds, average="macro")), 4) if rows else None}

    steps_per_epoch = (len(train_ex) + args.batch_size - 1) // args.batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps_per_epoch * args.epochs),
                                                steps_per_epoch * args.epochs)

    history = [{"epoch": 0, "val": evaluate(val_ex)}]
    print(f"  before training: val {history[0]['val']}")
    best, best_state = -1.0, copy.deepcopy(model.state_dict())
    t0 = time.perf_counter()
    model.train()
    for epoch in range(1, args.epochs + 1):
        running, seen = 0.0, 0
        for step, (enc, y) in enumerate(batches(train_ex, shuffle=True), start=1):
            loss = loss_fn(model(**enc).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            running += float(loss) * len(y); seen += len(y)
            if step % 25 == 0 or step == steps_per_epoch:
                el = time.perf_counter() - t0
                done = (epoch - 1) * steps_per_epoch + step
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {running / seen:.4f} "
                      f"elapsed {el:.0f}s eta {el / done * (steps_per_epoch * args.epochs - done):.0f}s", flush=True)
        val = evaluate(val_ex)
        history.append({"epoch": epoch, "train_loss": round(running / max(seen, 1), 4), "val": val})
        print(f"  epoch {epoch}: train loss {running / max(seen, 1):.4f}  val {val}")
        score = val["macro_f1"] if val["macro_f1"] is not None else -running
        if score > best:
            best, best_state = score, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    report = {
        "base_model": args.base_model, "output": str(out), "training_claims": len(train_claims),
        "examples": {"train": len(train_ex), "val": len(val_ex), **counts},
        "config": {k: v for k, v in vars(args).items() if k not in ("limit",)},
        "best_val_macro_f1": best, "history": history,
        "training_time_seconds": round(time.perf_counter() - t0, 1),
        "_provenance": "Produced by experiments/finetune_nli.py on the user's machine (CPU). "
                       "Validation is a held-out 10% of the TRAIN split; dev was not used.",
    }
    if not args.limit:
        Path("experiments/results").mkdir(parents=True, exist_ok=True)
        Path("experiments/results/finetune_nli.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {out}. Evaluate on dev with:\n  python -m experiments.run_proposed --retrieval bm25 "
          f"--no-sentence-selection --nli-model {out.as_posix()}")


if __name__ == "__main__":
    main()
