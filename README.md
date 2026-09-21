# SciVerify

**Open-Domain Scientific Claim Verification using Retrieval-Augmented Generation**

Research question: *Can an open-domain retrieval + evidence-verification
pipeline accurately determine whether a scientific claim is supported,
contradicted, or unresolved by scientific literature?*

Benchmark: [SciFact](https://github.com/allenai/scifact) (Wadden et al., 2020,
EMNLP) — 1,409 expert-written claims verified against a 5,183-abstract corpus
drawn from high-quality biomedical journals.

## Status: Phase 3 of 13 (see `experiments/ROADMAP.md`)

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

**Phase 5 — code ready, awaiting a real run:** Baseline C (retrieval + NLI
verification). Added:
- `verification/nli.py` — real MNLI-style classification (premise=evidence, hypothesis=claim) via `facebook/bart-large-mnli`, not a placeholder
- `verification/aggregate.py` — combines per-document NLI predictions into one claim verdict, with explicit CONFLICTING handling (spec section 12) rather than silently picking a side
- `evaluation/verification_metrics.py` — accuracy, per-class precision/recall/F1, macro-F1 via scikit-learn
- `experiments/run_baseline_c.py` — full pipeline: retrieve top-3 evidence docs (BM25 by default — see `configs/verification.yaml` for why dense isn't the default here), run NLI on each, aggregate to a verdict, score against `Claim.gold_verdict()`

29/29 tests pass, including 8 new tests directly exercising the conflicting-evidence rule. This is unit-tested with hand-built NLI predictions — the real `bart-large-mnli` model (~1.6GB) needs Hugging Face access this sandbox doesn't have. Run on your machine:
```powershell
pip install -r requirements.txt
pytest tests/ -v
python -m experiments.run_baseline_c
```
First run downloads the NLI model, then evaluates the full pipeline (all 300 dev claims, including NEI ones scored as UNKNOWN) and prints real retrieval + verification metrics. This will likely take longer than the previous runs — NLI inference on CPU is the slowest stage so far.

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
