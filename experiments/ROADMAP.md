# SciVerify development roadmap

Status is updated at the end of every phase with what actually works —
not what was attempted.

| Phase | Description | Status |
|---|---|---|
| 1 | Repo structure, environment, dataset schema/loader/download script | **Done** — code unit-tested (7/7) against synthetic fixtures; real dataset downloaded and verified by user: corpus=5183, train=809, dev=300, test=300 (sums to 1,409, matching the paper) |
| 2/3 | Dataset pipeline + BM25 baseline (Baseline A) | **Done** — real results on dev set: Recall@1=0.702, Recall@5=0.899, Recall@10=0.936, MRR=0.782 (188/300 dev claims scored; 112 excluded as NEI). See README for full table. |
| 4 | Dense retrieval baseline (Baseline B) | **Done** — real results: Recall@1=0.628, Recall@5=0.878, Recall@10=0.920, MRR=0.731 (all-MiniLM-L6-v2). **Underperforms BM25 on every metric** — reported honestly, see README for discussion. |
| 5 | Dense retrieval + NLI (Baseline C) | **Code done, unit-tested (29/29 total)** — awaiting real run on user's machine (needs huggingface.co access for bart-large-mnli, ~1.6GB) |
| 6 | NLI verification module | **Folded into Phase 5** (verification/nli.py + verification/aggregate.py) |
| 7 | Proposed hybrid pipeline (retrieval + rerank + NLI + conflict handling) | Not started |
| 8 | Benchmark experiments, real metrics | Not started |
| 9 | Ablation study | Not started |
| 10 | Error analysis | Not started |
| 11 | Streamlit UI + FastAPI backend | Not started |
| 12 | Docker + full test suite | Not started |
| 13 | Research report with real results | Not started |
