# SciVerify development roadmap

Status is updated at the end of every phase with what actually works —
not what was attempted.

| Phase | Description | Status |
|---|---|---|
| 1 | Repo structure, environment, dataset schema/loader/download script | **Done** — code unit-tested (7/7) against synthetic fixtures; real dataset downloaded and verified by user: corpus=5183, train=809, dev=300, test=300 (sums to 1,409, matching the paper) |
| 2/3 | Dataset pipeline + BM25 baseline (Baseline A) | **Done** — real results on dev set: Recall@1=0.702, Recall@5=0.899, Recall@10=0.936, MRR=0.782 (188/300 dev claims scored; 112 excluded as NEI). See README for full table. |
| 4 | Dense retrieval baseline (Baseline B) | **Done** — real results: Recall@1=0.628, Recall@5=0.878, Recall@10=0.920, MRR=0.731 (all-MiniLM-L6-v2). **Underperforms BM25 on every metric** — reported honestly, see README for discussion. |
| 5 | Retrieval + NLI (Baseline C) | **Done** — real results (bart-large-mnli, BM25 top-3): accuracy 0.590, macro-F1 0.450 (0.600 over classes with gold examples). See README. |
| 6 | Proposed system (hybrid retrieval + sentence selection + NLI + conflict-aware aggregation) | **Run on real data: accuracy 0.517 vs Baseline C 0.590 (worse).** Cause not yet isolated; ablation pending. |
| 7 | (merged into Phase 6) | — |
| 8 | Benchmark experiments, real metrics | Not started |
| 9 | Ablation study | Not started |
| 10 | Error analysis | Not started |
| 11 | Streamlit UI + FastAPI backend | Not started |
| 12 | Docker + full test suite | Not started |
| 13 | Research report with real results | Not started |
