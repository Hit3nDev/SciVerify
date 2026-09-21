# SciVerify development roadmap

Status is updated at the end of every phase with what actually works —
not what was attempted.

| Phase | Description | Status |
|---|---|---|
| 1 | Repo structure, environment, dataset schema/loader/download script | **Done** — code unit-tested (7/7) against synthetic fixtures; real dataset downloaded and verified by user: corpus=5183, train=809, dev=300, test=300 (sums to 1,409, matching the paper) |
| 2/3 | Dataset pipeline + BM25 baseline (Baseline A) | **Done** — real results on dev set: Recall@1=0.702, Recall@5=0.899, Recall@10=0.936, MRR=0.782 (188/300 dev claims scored; 112 excluded as NEI). See README for full table. |
| 4 | Dense retrieval baseline (Baseline B) | **Code done, unit-tested (17/17 total)** — awaiting real run on user's machine (needs huggingface.co access for model download) |
| 5 | Dense retrieval + NLI (Baseline C) | Not started |
| 6 | NLI verification module | Not started |
| 7 | Proposed hybrid pipeline (retrieval + rerank + NLI + conflict handling) | Not started |
| 8 | Benchmark experiments, real metrics | Not started |
| 9 | Ablation study | Not started |
| 10 | Error analysis | Not started |
| 11 | Streamlit UI + FastAPI backend | Not started |
| 12 | Docker + full test suite | Not started |
| 13 | Research report with real results | Not started |
