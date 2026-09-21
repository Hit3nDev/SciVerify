# SciVerify

**Open-Domain Scientific Claim Verification using Retrieval-Augmented Generation**

Research question: *Can an open-domain retrieval + evidence-verification
pipeline accurately determine whether a scientific claim is supported,
contradicted, or unresolved by scientific literature?*

Benchmark: [SciFact](https://github.com/allenai/scifact) (Wadden et al., 2020,
EMNLP) — 1,409 expert-written claims verified against a 5,183-abstract corpus
drawn from high-quality biomedical journals.

## Status: Phase 1 of 13 (see `experiments/ROADMAP.md`)

**Done and verified in this environment:**
- Repository skeleton (`sciverify/{data,models,retrieval,verification,evaluation,experiments,api,frontend,tests,configs,notebooks}`)
- `configs/data.yaml` — all dataset paths/labels/seeds centralized, nothing hardcoded in code
- `data/schema.py` — typed dataclasses matching SciFact's documented schema exactly (claim, evidence keyed by doc_id, rationale sentences/labels)
- `data/download_data.py` — reproducible download+extract script
- `data/loader.py` — JSONL loader, label-count summary, conflicting-evidence detection
- `tests/test_data_loader.py` — **7/7 passing** against synthetic fixtures shaped exactly like real SciFact rows

**Known limitation, stated plainly:** the sandbox this was built in only
allows outbound network access to a small domain whitelist (PyPI, npm,
GitHub, etc.) and cannot reach `scifact.s3-us-west-2.amazonaws.com` or
`huggingface.co`. I confirmed this with a direct request (403,
`host_not_allowed`) rather than assuming. That means:

- `download_data.py` is written correctly and will work on a normal machine, but **has not been run against the real dataset yet**, and
- **no real dataset statistics, retrieval numbers, or verification metrics exist yet.** Every number in this README, in future phases, and eventually in the research report will be the actual output of a real run — not estimated or recalled from training data.

### To unblock Phase 1 fully
Run, on any machine with normal internet access:
```bash
pip install -r requirements.txt
python -m data.download_data
python -c "from pathlib import Path; from data.loader import load_config, dataset_summary; c = load_config(); print(dataset_summary(Path(c['paths']['raw_dir']), c))"
```
This will print the real corpus size, claim counts per split, and label
distribution. Paste that output back and I will use it verbatim to correct
any numbers, then continue to Phase 2 (BM25 baseline) with confidence the
foundation is right.

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
