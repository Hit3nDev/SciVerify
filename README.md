# SciVerify

**Open-Domain Scientific Claim Verification using Retrieval-Augmented Generation**

Research question: *Can an open-domain retrieval + evidence-verification
pipeline accurately determine whether a scientific claim is supported,
contradicted, or unresolved by scientific literature?*

Benchmark: [SciFact](https://github.com/allenai/scifact) (Wadden et al., 2020,
EMNLP) — 1,409 expert-written claims verified against a 5,183-abstract corpus
drawn from high-quality biomedical journals.

## Status: Phase 2 of 13 (see `experiments/ROADMAP.md`)

**Real dataset confirmed** (downloaded and verified by the user on 2026-09-21,
not estimated or recalled from training data):

| File | Line count |
|---|---|
| `corpus.jsonl` | 5,183 |
| `claims_train.jsonl` | 809 |
| `claims_dev.jsonl` | 300 |
| `claims_test.jsonl` | 300 |

809 + 300 + 300 = 1,409 total claims, matching the count reported in
Wadden et al. 2020. This is the first real, verified checkpoint in the
project — everything from here on is built and evaluated against these
actual files.

**Phase 1 — done:**
- Repository skeleton, `configs/data.yaml`, typed schema, download script, JSONL loader
- `tests/test_data_loader.py` — 7/7 passing against synthetic fixtures
- Real download executed and verified by the user (sandbox here still cannot
  reach the S3 bucket directly — see `data/download_data.py` docstring —
  so real-data runs happen on the user's machine; all code is written to
  work identically either way)

**Phase 2 — in progress:** BM25 sparse retrieval baseline (Baseline A).

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
