# SciVerify development roadmap

Status is updated at the end of every phase with what actually works —
not what was attempted.

| Phase | Description | Status |
|---|---|---|
| 1 | Repo structure, environment, dataset schema/loader/download script | **Done** — code unit-tested (7/7) against synthetic fixtures; real dataset downloaded and verified by user: corpus=5183, train=809, dev=300, test=300 (sums to 1,409, matching the paper) |
| 2/3 | Dataset pipeline + BM25 baseline (Baseline A) | **Done** — real results on dev set: Recall@1=0.702, Recall@5=0.899, Recall@10=0.936, MRR=0.782 (188/300 dev claims scored; 112 excluded as NEI). See README for full table. |
| 4 | Dense retrieval baseline (Baseline B) | **Done** — real results: Recall@1=0.628, Recall@5=0.878, Recall@10=0.920, MRR=0.731 (all-MiniLM-L6-v2). **Underperforms BM25 on every metric** — reported honestly, see README for discussion. |
| 5 | Retrieval + NLI (Baseline C) | **Done** — real results (bart-large-mnli, BM25 top-3): accuracy 0.590, macro-F1 0.450 (0.600 over classes with gold examples). See README. |
| 6 | Proposed system (hybrid + sentence selection + NLI + conflict-aware aggregation) and first ablation | **Done, real results: proposed system 0.517 accuracy vs Baseline C 0.590. Ablation shows sentence selection (n=2) is the cause of the drop; hybrid retrieval gives only a small retrieval gain.** |
| 7 | (merged into Phase 6) | — |
| 8 | Full benchmark experiments | **Done for the current systems** (A, B, C, hybrid, sentence-selection variants, threshold sweep). Timings are invalid and must be re-measured. |
| 9 | Ablation study | **Done for the components built:** sentence selection hurts, hybrid retrieval gives a small retrieval gain only, threshold tuning gives nothing. Single runs, no significance test. |
| 10 | Error analysis | **Tool built and tested on synthetic data (`experiments/error_analysis.py`); not yet run on real predictions.** |
| 11 | Streamlit UI + FastAPI backend | Not started |
| 12 | Docker + full test suite | Not started |
| 13 | Research report with real results | Not started |
