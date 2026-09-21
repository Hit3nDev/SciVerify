# Data License Notice

SciVerify's **code** is MIT-licensed. The **SciFact dataset** it is built and
evaluated on is a separate artifact with its own license:

- License: **CC BY-NC 2.0** (Creative Commons Attribution-NonCommercial 2.0)
- Source: AllenAI, https://github.com/allenai/scifact
- Paper: Wadden et al., "Fact or Fiction: Verifying Scientific Claims", EMNLP 2020

By downloading SciFact via `data/download_data.py` you agree to AllenAI's usage
terms. In particular: **non-commercial use only**. If you build on SciVerify
for a commercial product, you must either obtain a separate license for the
underlying data or retrain/evaluate on data you're licensed to use commercially.

No dataset files are committed to this repository. `download_data.py` fetches
them at run time from AllenAI's official distribution.
