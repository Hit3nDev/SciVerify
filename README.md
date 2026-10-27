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
| Total time / per claim | not reliable (see timing note) |

**How to read this honestly:**
- The two macro-F1 numbers differ only because the dev set has no CONFLICTING gold claims, so that class can only score 0. 0.600 is the fairer summary of verification quality; 0.450 is reported too so nothing is hidden. The 0.600 figure was derived afterwards from the per-class F1s, not printed by the run.
- SUPPORTED predictions are mostly right (precision 0.81) but many supported claims are missed (recall 0.56). CONTRADICTED is the opposite: recall 0.66, precision 0.45, so the NLI model over-predicts contradiction.
- Only 65 of 112 NEI claims were labelled UNKNOWN. Retrieval always returns 3 documents, so there is usually something for the NLI model to react to.
- At least one claim was predicted CONFLICTING, and every such prediction counts as an error because no dev claim is conflicting. Detecting conflict is a design choice this benchmark cannot reward.
- This run used `bart-large-mnli`, not the lighter MiniLM model now in `configs/verification.yaml`. Later comparisons must either use the same NLI model or rerun this baseline with the new one.
- Retrieval numbers are identical to Baseline A because Baseline C reuses BM25.
- This is a single run on the dev set with no confidence intervals or repeated seeds.

**Phase 6 + ablation — real results. The proposed system does not beat Baseline C.**
All four rows: same 300 dev claims, same NLI model (`bart-large-mnli`), top-3 documents, one run each.

| Retrieval | Sentence selection | Correct (of 300) | Accuracy | Macro-F1 (gold classes) | Recall@10 |
|---|---|---|---|---|---|
| BM25 | no (Baseline C) | 177 | **0.590** | **0.600** | 0.936 |
| Hybrid | no | 174 | 0.580 | 0.592 | 0.947 |
| Hybrid | yes, 2 sentences (proposed) | 155 | 0.517 | 0.527 | 0.947 |
| BM25 | yes, 2 sentences | 149 | 0.497 | 0.509 | 0.936 |
| BM25 | yes, 4 sentences | 169 | 0.563 | 0.572 | 0.936 |
| BM25 | yes, 5 sentences | 167 | 0.557 | 0.568 | 0.936 |

**Evidence-selection diagnostic** (`eval_sentence_selection.py`, 209 claim/gold-document pairs): how often the chosen sentences include an annotated evidence sentence.

| Sentences kept (n) | Hit rate, my selection | Hit rate, first n sentences | Share of gold sentences found |
|---|---|---|---|
| 1 | 0.464 | 0.057 | 0.327 |
| 2 | 0.679 | 0.148 | 0.509 |
| 3 | 0.799 | 0.263 | 0.651 |
| 4 | 0.880 | 0.411 | 0.735 |
| 5 | 0.919 | 0.569 | 0.795 |

**What the ablation shows:**
- **Sentence selection with n=2 lowered verification accuracy in both matched comparisons:** BM25 0.590 to 0.497 (28 fewer claims correct) and hybrid 0.580 to 0.517 (19 fewer). The drop is mostly SUPPORTED recall (0.565 to 0.379 for BM25).
- **A plausible mechanism, supported by the diagnostic:** with n=2 the chosen sentences contain an annotated evidence sentence only 68% of the time, so about a third of the time the NLI model never sees the sentence that supports the claim. The selection heuristic is far better than taking the first n sentences, but not good enough. This is consistent with the drop; it is not proof that it explains all of it.
- **Hybrid retrieval raised retrieval metrics slightly** (Recall@5 0.899 to 0.915, Recall@10 0.936 to 0.947, MRR 0.782 to 0.794, Recall@1 0.702 to 0.697) **but did not improve verification:** 0.580 vs 0.590, which is 3 claims and well within what one run can vary by.
- **Best configuration so far is still Baseline C.** No component tested has improved verification accuracy.
- **Speed:** no speed claim is made; the available timings are not trustworthy (see the timing note below).

**Limits of these results:** one run per configuration, no repeated seeds, no confidence intervals and no paired significance test (per-claim predictions were not saved). Differences of a few claims should not be read as real. The 9-point and 6-point drops from selection are larger but have not been tested statistically.

**Sentences-per-document sweep (BM25, same setup):** accuracy was 0.497 at n=2, 0.563 at n=4, 0.557 at n=5, and 0.590 with the whole abstract. It rises toward the baseline as more context is kept and never exceeds it. The 4-vs-5 difference is 2 claims, i.e. noise. So at no setting tested does selection beat feeding the whole abstract.

**Timing note (important):** wall-clock timings from these runs are NOT valid and are not used anywhere in this project's conclusions. The n=4 and n=5 runs were slower than the whole-abstract baseline despite reading less text; the train run's log jumps from 5,395 s to 32,678 s and later to 61,674 s with no work in between (the laptop was suspended); and the identical Baseline C configuration took 6.7 s/claim in one run and 41.7 s/claim in another. Accuracy and F1 are unaffected. Inference latency must be re-measured under controlled conditions (median per-call time, same session) before it is reported.

**Confidence-threshold tuning: no gain (real result).** The NLI outputs were saved for all 809 train and 300 dev claims (BM25, whole abstract, `bart-large-mnli`), the threshold was chosen on train only, then applied to dev once.

| Threshold | Train acc | Train macro-F1 | Dev acc | Dev macro-F1 |
|---|---|---|---|---|
| 0.40 | 0.557 | 0.577 | 0.580 | 0.592 |
| 0.50 (default) | 0.564 | 0.580 | 0.590 | 0.600 |
| 0.60 | 0.582 | 0.589 | 0.593 | 0.593 |
| **0.70 (chosen on train)** | 0.596 | 0.593 | 0.587 | 0.580 |
| 0.80 | 0.587 | 0.578 | 0.607 | 0.592 |
| 0.85 | 0.582 | 0.571 | 0.593 | 0.576 |
| 0.90 | 0.572 | 0.554 | 0.540 | 0.516 |
| 0.95 | 0.549 | 0.515 | 0.513 | 0.477 |
| 0.98 | 0.501 | 0.442 | 0.487 | 0.428 |

- The train-chosen threshold (0.70) gives dev accuracy 0.587 vs 0.590 at the default: no improvement. The small in-sample gain on train (macro-F1 0.580 to 0.593) did not transfer.
- Accuracy is flat (about 0.56 to 0.61) for thresholds 0.4 to 0.85 and only falls at 0.90 and above, where too many claims become UNKNOWN. The dev peak at 0.80 (0.607) is not used: choosing it would be tuning on dev, and it is within noise of its neighbours.
- So the cutoff is not what limits accuracy. The limit is the NLI model's ability to tell evidence from related-but-non-evidence text, and the decision rule built on top of it.
- Baseline C on the 809 train claims scores 0.564 accuracy / 0.580 macro-F1, slightly below the dev figures (0.590 / 0.600). The larger sample is the better estimate of its true accuracy; the dev set (300 claims, standard error roughly 3 points) is small, so dev differences of a few points between configurations should not be over-read.
- Reproducibility: re-running Baseline C on dev reproduced 0.590 exactly.

**Error analysis (Phase 10) — real results for Baseline C on the dev set: 123 of 300 verdicts wrong (41%).**

| Gold \ predicted | SUPPORTED | CONTRADICTED | UNKNOWN | CONFLICTING |
|---|---|---|---|---|
| SUPPORTED (124) | 70 | 17 | 25 | 12 |
| CONTRADICTED (64) | 7 | 42 | 15 | 0 |
| UNKNOWN (112) | 9 | 34 | 65 | 4 |

| Automatic error category | Errors | Share |
|---|---|---|
| Verdict given although no gold evidence exists | 43 | 35% |
| Gold evidence retrieved, judge did not recognise it | 27 | 22% |
| Gold evidence retrieved, judge said the opposite | 22 | 18% |
| Spurious CONFLICTING (dev has no conflicting claims) | 16 | 13% |
| Gold evidence not among the 3 retrieved papers | 15 | 12% |

What the data shows (measured), and what is only a hypothesis (marked):
1. **88% of errors (108/123) happen after retrieval**, in the judging or decision stage. Retrieval misses are 12%. This agrees with the earlier finding that retrieval is already good (Recall@10 = 0.936).
2. **The judge often mislabels the key paper itself.** Of 158 claims with gold evidence where the annotated paper was retrieved, the judge labelled that paper correctly for 104 (65.8%); it said neutral for 36 and the opposite for 18. Another 12 claims were lost afterwards although the key paper was judged correctly. A further 30 claims had the annotated paper outside the top 3.
3. **No-evidence claims get a verdict too often:** 43 of 112 received SUPPORTED or CONTRADICTED, 34 of them CONTRADICTED (30% of such claims, versus 8% SUPPORTED). *Hypothesis, not tested:* MNLI-style models tend to answer "contradiction" for text that is on-topic but does not address the claim, and SciFact's no-evidence claims are on-topic for the retrieved papers. The median confidence of these wrong verdicts is 0.78 versus 0.89 for correct SUPPORTED/CONTRADICTED verdicts, an overlap that is consistent with why the confidence cutoff could not separate them.
4. **16 spurious conflicts** (12 SUPPORTED, 4 no-evidence claims): typically one paper judged entailment with high confidence and another judged contradiction. The saved outputs cannot tell whether the contradicting paper is a genuine disagreement or merely a loosely related paper; that needs reading the abstracts.
5. **Claim properties show no detectable effect:** numeric claims 41.2% errors vs 40.9%; negated claims 40.0% vs 41.1%; claims longer than the median 39.5% vs 42.1%. The flags are crude (any digit, a negation word list), so this does not prove these are easy. Only 10 dev claims need two or more evidence papers (2 errors), too few to conclude anything.
6. **Ceiling for decision rules:** even a rule with perfect knowledge of which claims have evidence and which paper to trust, using only the judge's saved labels, could reach 0.79 (237/300). That bound is optimistic; a realistic rule would gain far less. For 63 of the 188 gold-evidence claims, none of the three retrieved papers received the right label, and no decision rule can fix those. Larger gains therefore need a better judge (or better retrieval for the 30 misses), not a different way of combining the same judgements.

**Coverage of the error types in the project specification:** measured automatically: retrieval failure, conflicting evidence (as spurious conflict), insufficient evidence (verdict without gold evidence), numerical claims (crude), multi-evidence claims (too few). Not measurable automatically and **not yet analysed**: terminology mismatch, ambiguous claims, misleading evidence, near-duplicate irrelevant papers. `error_analysis.py` writes a CSV with a blank `manual_category` column for labelling these by reading the abstracts.

**Not yet tried:** an NLI model trained on scientific claims (for example fine-tuned on the SciFact train rationales), and a different decision rule for combining per-document judgements.

## Run the demo (Phase 11)

```powershell
uvicorn api.main:app --port 8000                                                                 # web app at http://localhost:8000, API docs at /docs
python -m experiments.demo "Monoclonal antibody targeting of N-cadherin inhibits metastasis."   # command line
streamlit run frontend/app.py                                                                    # older, simpler web page
```
The demo runs the best system found in the evaluation (BM25 + whole-abstract NLI with `facebook/bart-large-mnli`). See `DEMO.md` for the full guide.

The web app (`frontend/web/`, plain HTML/CSS/JS served by FastAPI, works offline) has five sections:

- **Verify:** live progress as each paper is read, then the verdict, a weight-of-evidence bar, and one card per paper. Each card shows its search score, the model's full entail/neutral/contradict split, and the abstract with the best-matching sentences and claim words highlighted. *Find deciding sentences* judges every sentence on its own (a reading aid only: the verdict always uses the whole abstract). Dataset claims show the dataset's answer and the annotated evidence paper. Settings change the number of papers and the evidence threshold. Results can be shared as a link or exported as Markdown, JSON or PDF. Recent checks are kept in the browser.
- **Batch:** many claims at once (typed, imported from CSV/TXT, or drawn from the dev set), with live agreement against the answer key and CSV/JSON export.
- **Library:** BM25 search over the 5,183 abstracts, plus a paper reader.
- **Benchmarks:** charts drawn from `experiments/results/*.json`.
- **Method:** how a verdict is reached and the system's limits.

API endpoints: `POST /verify` (optional `top_k`, `min_confidence`), `POST /verify/stream` (newline-delimited JSON progress events), `POST /explain`, `GET /search`, `GET /papers/{id}`, `GET /examples`, `GET /benchmarks`, `GET /status`. Repeated claims are answered from an in-memory cache.

**Status:** covered by 94 passing tests that use stand-in models. The real-model path was also run end to end through the web app. Latency is not reported (timings were invalid).

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
├── frontend/          # web/ = the web app served by the API; app.py = older Streamlit page
├── tests/             # pytest suite, growing with each phase
├── configs/           # data.yaml, model.yaml, experiment configs
└── notebooks/         # (Phase 10) error analysis
```
