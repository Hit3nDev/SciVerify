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

**Phase 1–3 — done** (see checkpoints above and in ROADMAP.md).
**Phase 4 — code ready, awaiting a real run:** dense retrieval baseline
(Baseline B) — `retrieval/embeddings.py`, `retrieval/dense.py` (FAISS flat
IP index), `experiments/run_dense_baseline.py`. Unit-tested (3/3 new tests,
17/17 total) against a hand-crafted fake embedder, since loading the real
`sentence-transformers/all-MiniLM-L6-v2` model requires downloading it from
huggingface.co, which this sandbox can't reach. Run on your machine:
```powershell
pip install -r requirements.txt
pytest tests/ -v
python -m experiments.run_dense_baseline
```
The first run downloads the ~80MB MiniLM model, then builds a FAISS index
and evaluates Recall@K/MRR on the real dev set. Paste back the printed
JSON and I'll record it next to the BM25 numbers for a real comparison.

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
