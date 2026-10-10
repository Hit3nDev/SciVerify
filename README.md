# SciVerify

**Open-Domain Scientific Claim Verification using Retrieval-Augmented Generation**

Research question: *Can an open-domain retrieval + evidence-verification
pipeline accurately determine whether a scientific claim is supported,
contradicted, or unresolved by scientific literature?*

Benchmark: [SciFact](https://github.com/allenai/scifact) (Wadden et al., 2020,
EMNLP) — 1,409 expert-written claims verified against a 5,183-abstract corpus
drawn from high-quality biomedical journals.

## Status: Phase 5 of 13 (see `experiments/ROADMAP.md`)

**Real BM25 baseline results** (run by the user on 2026-09-21, on the real
downloaded dev set — not simulated):

| Metric | Value |
|---|---|
| Corpus size | 5,183 |
| Dev claims total | 300 |
| Dev claims scored (have gold evidence) | 188 |
| Dev claims excluded (NEI, no evidence to retrieve) | 112 |
| Recall@1 | 0.702 |
| Recall@5 | 0.899 |
| Recall@10 | 0.936 |
| MRR | 0.782 |
| Index build time | 0.90s |
| Avg latency/claim | 35.9ms |

Config: `simple_lowercase_alnum` tokenizer, k1=1.5, b=0.75, top_k=10 (see
`configs/retrieval.yaml`). Full output in `experiments/results/bm25_baseline.json`.

**Interpretation, not spin:** BM25 does well here because SciFact claims are
written by rewriting a citation sentence from the source abstract, so
vocabulary overlap with the true evidence document is naturally high. This
is a real property of the dataset, not a property of retrieval in general —
worth calling out explicitly in the final report's Limitations section,
since a harder open-domain corpus (e.g. SciFact-Open) would likely show a
bigger gap between BM25 and dense/hybrid retrieval. That's exactly the kind
of comparison Phase 4 (dense) and Phase 7 (hybrid) will test.

**Phase 1–4 — done, with real comparative results:**

| Metric | BM25 (Baseline A) | Dense MiniLM (Baseline B) |
|---|---|---|
| Recall@1 | **0.702** | 0.628 |
| Recall@5 | **0.899** | 0.878 |
| Recall@10 | **0.936** | 0.920 |
| MRR | **0.782** | 0.731 |
| Index build time | 0.90s | 232.9s |
| Avg latency/claim | 35.9ms | 13.1ms |

**Honest finding, not spun:** BM25 beats dense retrieval on every recall
metric here. This is consistent with the hypothesis noted after the BM25
run — SciFact claims are rewritten citation sentences, so they share a lot
of vocabulary with their source abstract, which plays directly to lexical
retrieval's strength and doesn't require semantic generalization. A
general-purpose sentence embedding model (MiniLM, not domain-tuned on
biomedical text) has no particular advantage here and pays a real cost in
index-build time. This is exactly the kind of result section 6 (ablation)
and the final report's Limitations section are supposed to contain — a
real baseline losing to a simpler one, reported as-is rather than hidden
or re-run until it looks better.

This raises a concrete, testable hypothesis for later phases: hybrid
retrieval (Phase 7) may only pay off if it's dominated by the sparse
signal, or if a biomedical-domain embedding model (e.g. PubMedBERT-based)
is used instead of general-purpose MiniLM. Worth an ablation once Phase 7
exists rather than assuming either way now.

**Phase 5 — done, with real results (Baseline C: BM25 retrieval + NLI):**

Run on a CPU-only Windows machine, all 300 dev claims, `facebook/bart-large-mnli`,
BM25 top-3 evidence, confidence threshold 0.5.

| Class | Gold claims | Precision | Recall | F1 |
|---|---|---|---|---|
| SUPPORTED | 124 | 0.814 | 0.565 | 0.667 |
| CONTRADICTED | 64 | 0.452 | 0.656 | 0.535 |
| UNKNOWN (NEI) | 112 | 0.619 | 0.580 | 0.599 |
| CONFLICTING | 0 | 0.000 | 0.000 | 0.000 |

| Metric | Value |
|---|---|
| Accuracy | 0.590 (177/300) |
| Macro-F1, all four labels | 0.450 |
| Macro-F1, labels with gold examples only | 0.600 (derived) |
| Total time / per claim | 2,006 s / 6.7 s |

**How to read this honestly:**
- The two macro-F1 numbers differ only because the dev set has no CONFLICTING gold claims, so that class can only score 0. 0.600 is the fairer summary of verification quality; 0.450 is reported too so nothing is hidden. The 0.600 figure was derived afterwards from the per-class F1s, not printed by the run.
- SUPPORTED predictions are mostly right (precision 0.81) but many supported claims are missed (recall 0.56). CONTRADICTED is the opposite: recall 0.66, precision 0.45, so the NLI model over-predicts contradiction.
- Only 65 of 112 NEI claims were labelled UNKNOWN. Retrieval always returns 3 documents, so there is usually something for the NLI model to react to.
- At least one claim was predicted CONFLICTING, and every such prediction counts as an error because no dev claim is conflicting. Detecting conflict is a design choice this benchmark cannot reward.
- This run used `bart-large-mnli`, not the lighter MiniLM model now in `configs/verification.yaml`. Later comparisons must either use the same NLI model or rerun this baseline with the new one.
- Retrieval numbers are identical to Baseline A because Baseline C reuses BM25.
- This is a single run on the dev set with no confidence intervals or repeated seeds.

**Phase 6 — proposed system: code ready, NOT yet run on real data.**
Baseline C's weak spot is verification, not retrieval (BM25 already reaches Recall@10 = 0.936), so the proposed system changes the judging step:
- `retrieval/hybrid.py` — fuses BM25 and dense rankings with Reciprocal Rank Fusion
- `verification/evidence_selection.py` — gives the NLI model only the best-matching sentences instead of the whole abstract
- `experiments/run_proposed.py` — runs the full system or any ablation of it via flags

38/38 tests pass (synthetic fixtures). There is no accuracy claim for this system yet; the first real run decides whether it helps. It may not.

Fair comparison with Baseline C requires the same NLI model:
```powershell
python -m experiments.run_proposed --nli-model facebook/bart-large-mnli --retrieval bm25 --limit 30   # quick check
python -m experiments.run_proposed --nli-model facebook/bart-large-mnli                             # full proposed system
python -m experiments.run_proposed --nli-model facebook/bart-large-mnli --no-sentence-selection     # ablation
```
Not yet attempted: tuning the confidence threshold (must be tuned on train claims, not dev) and a science-specific NLI model.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

## License
Code: MIT. **Dataset: CC BY-NC 2.0 (non-commercial)** — see
`data/DATA_LICENSE_NOTICE.md`. This is AllenAI's license on SciFact, not a
choice made by this project.

## Repository structure
```
sciverify/
├── data/            # schema, loader, download script, license notice
├── models/          # (Phase 5+) NLI / embedding model wrappers
├── retrieval/        # (Phase 4+) BM25, dense, hybrid retrieval
├── verification/      # (Phase 6+) NLI verification + conflict aggregation
├── evaluation/        # (Phase 8+) metrics: Recall@K, MRR, F1, Macro-F1
├── experiments/       # experiment runner, ROADMAP.md, results (once real)
├── api/               # (Phase 11) FastAPI backend
├── frontend/          # (Phase 11) Streamlit UI
├── tests/             # pytest suite, growing with each phase
├── configs/           # data.yaml, model.yaml, experiment configs
└── notebooks/         # (Phase 10) error analysis
```
