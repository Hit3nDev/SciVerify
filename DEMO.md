# SciVerify: demo and usage guide

## Tonight (about 15 minutes, do this once)

```powershell
cd SciVerify
.venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -q                      # expect: 94 passed
python -m experiments.demo "Monoclonal antibody targeting of N-cadherin inhibits metastasis."
```

The last command is the real test. It loads the corpus and the language model (the first run can take a minute or more), retrieves three papers, judges each one and prints a verdict. **If it prints a verdict, everything works.** It also warms the model so tomorrow is faster.

Then start the web app once and try one example:

```powershell
uvicorn api.main:app --port 8000
```

Open http://localhost:8000. The status pill (top right) turns green when the model is loaded.
(The older Streamlit page still works as a fallback: `streamlit run frontend/app.py`.)

If the classroom has no internet, run this in PowerShell before starting, so the program does not wait on the network (the model must already be downloaded, which tonight's run does):

```powershell
$env:HF_HUB_OFFLINE = "1"
```

## Tomorrow: the order to show things (5 to 7 minutes)

1. **The problem (30 s).** "Anyone can claim anything about science. This tool checks a claim against real paper abstracts and shows which papers it used."
2. **Live web app.** Run `uvicorn api.main:app --port 8000`, open http://localhost:8000 and click an example card. The progress bar shows each step as it happens (search, then reading each paper, then weighing), so narrate it: "it searches 5,183 abstracts, then a language model reads the top 3 papers and decides whether each agrees."
   - `Monoclonal antibody targeting of N-cadherin inhibits metastasis.` (dataset says SUPPORTED; the page shows "✓ match" and marks the annotators' evidence paper)
   - `Chenodeosycholic acid treatment reduces whole-body energy expenditure.` (dataset says CONTRADICTED)
   - On a paper card, point at the **judge's reading** bar (entails / neutral / contradicts), then click **Find deciding sentences** to show which sentence decided it.
   - **Random dataset claim** draws a real dev claim whose answer you have not seen.
   - Optional: the **Benchmarks** tab draws every number below as charts, and **Batch** → draw 5 claims → Run shows a live mini-benchmark.
3. **Show a failure on purpose.** `ALDH1 expression is associated with poorer prognosis in breast cancer.` The dataset says SUPPORTED, but in our evaluation run the system said CONFLICTING. Its opposite, `ALDH1 ... better breast cancer outcomes`, is correctly CONTRADICTED. This shows you know where it breaks.
4. **Usage as a service (API).**
   ```powershell
   uvicorn api.main:app --port 8000
   ```
   The same server is the API. Open http://localhost:8000/docs in a browser (interactive documentation), or from a second terminal:
   ```powershell
   Invoke-RestMethod -Method Post -Uri http://localhost:8000/verify -ContentType "application/json" -Body '{"claim":"Monoclonal antibody targeting of N-cadherin inhibits metastasis."}' | ConvertTo-Json -Depth 5
   ```
   The first request is slow because the model loads then.
5. **The evidence it works and where it stops (README tables).**
   - Search finds the right paper in the top 10 for 93.6% of claims (BM25).
   - The final verdict is right for 59.0% of 300 dev claims (macro-F1 0.60 over the classes that occur).
   - I tried three improvements; none beat the simple version. The error analysis shows the weak part is the language model that judges papers (it labels the key paper correctly 65.8% of the time), not the search.
6. **Quality of the code.** Run `pytest tests/ -q` and show `94 passed`.

## What to say about the 59%

Say it first, plainly: "It is right about 6 times in 10 on a standard benchmark. I measured that honestly, tested ideas to improve it, and found out why it fails. The next step is to train the judging model on science claims." Do not say "accurate", "state of the art", or "deployed".

## Likely questions, honest answers

- **Why not just ask ChatGPT?** A chatbot answers from memory and shows no sources. This tool must retrieve papers and show them, and I can measure how often it is right on a benchmark.
- **Can I type any claim?** Yes, but it only searches 5,183 biomedical abstracts. Claims outside that area usually return UNKNOWN. That is a limit of the corpus.
- **Is the confidence number the chance it is right?** No. It is the language model's probability and it is not calibrated. In the evaluation, wrong verdicts on no-evidence claims had a median confidence of 0.78.
- **Why did keyword search beat the AI-embedding search?** SciFact claims are rewritten sentences from the source abstracts, so they share many words with the right paper, which favours keyword matching. This is a property of the dataset.
- **What would you improve?** Train the judging model on the 809 labelled training claims, since it has never seen science claims, then re-measure with a significance test.
- **How long does a claim take?** Tens of seconds on my laptop CPU. I do not have a reliable timing measurement, because the laptop went to sleep during the recorded runs.

## If something breaks during the demo

- Web page slow or stuck: switch to the command line (`python -m experiments.demo "..."`) or show the README results.
- Nothing runs: show `pytest tests/ -q`, the results tables in `README.md`, and the files in `experiments/results/`. These are real recorded outputs.
