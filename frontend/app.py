"""SciVerify web page.   streamlit run frontend/app.py

The examples below are real SciFact dev-set claims. "Dataset answer" is the label the dataset's
annotators gave. In the evaluation run, the system got the first five right and the last one wrong;
it is included on purpose so the limits are visible.
"""

import streamlit as st

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification import factory

EXAMPLES = [
    ("Neutrophil extracellular traps (NETs) are released by ANCA-stimulated neutrophils.", "SUPPORTED"),
    ("Monoclonal antibody targeting of N-cadherin inhibits metastasis.", "SUPPORTED"),
    ("Chenodeosycholic acid treatment reduces whole-body energy expenditure.", "CONTRADICTED"),
    ("ALDH1 expression is associated with better breast cancer outcomes.", "CONTRADICTED"),
    ("Flexible molecules experience greater steric hindrance in the tumor microenviroment than rigid molecules.",
     "UNKNOWN (no evidence in the dataset)"),
    ("ALDH1 expression is associated with poorer prognosis in breast cancer.", "SUPPORTED  (the system gets this one wrong)"),
]
COLORS = {"SUPPORTED": st.success, "CONTRADICTED": st.error, "CONFLICTING": st.warning, "UNKNOWN": st.info}


@st.cache_resource(show_spinner="Loading the corpus and the language model (first time only)...")
def get_verifier():
    return factory.build_verifier()


st.set_page_config(page_title="SciVerify", layout="centered")
st.title("SciVerify")
st.caption("Checks a scientific claim against retrieved paper abstracts and shows its evidence.")

with st.sidebar:
    st.header("Options")
    debug = st.checkbox("Research / debug mode", help="Show retrieval scores, raw NLI outputs and the aggregation step.")
    st.markdown("**Examples** (real dev-set claims)")
    labels = ["(type my own)"] + [c for c, _ in EXAMPLES]
    choice = st.selectbox("Pick an example", labels, label_visibility="collapsed")
    st.caption("The system searches only 5,183 biomedical abstracts, so claims outside that area "
               "will usually come back UNKNOWN.")

default = "" if choice == labels[0] else choice
claim = st.text_area("Scientific claim", value=default, height=90, placeholder="e.g. Monoclonal antibody targeting of N-cadherin inhibits metastasis.")
if choice != labels[0]:
    st.caption("Dataset answer: " + dict(EXAMPLES)[choice])

if st.button("Verify Claim", type="primary"):
    if not claim.strip():
        st.warning("Please enter a claim first.")
    else:
        with st.spinner("Retrieving papers and judging each one (this can take 20-60 s on a laptop CPU)..."):
            resp = get_verifier().verify(claim)

        COLORS.get(resp.verdict, st.info)(f"VERDICT: {resp.verdict}")
        c1, c2 = st.columns(2)
        c1.metric("Model confidence", f"{resp.confidence:.2f}")
        c2.metric("Processing time", f"{resp.elapsed_seconds:.1f} s")
        st.progress(min(max(resp.confidence, 0.0), 1.0))
        st.caption("Confidence is the language model's probability, not a calibrated chance of being right.")

        st.subheader("Why")
        st.write(resp.explanation)

        st.subheader("Evidence")
        if not resp.evidence:
            st.write("No papers were found for this claim.")
        for i, e in enumerate(resp.evidence, start=1):
            with st.expander(f"Paper {i}: {e.title}  ->  {e.judgement}", expanded=(i == 1)):
                st.write(e.passage)
                st.caption(f"Paper ID {e.doc_id} | the judge read the whole abstract; the sentences above are the ones that best match the claim.")

        if debug:
            st.subheader("Research / debug details")
            st.table([{"paper": e.doc_id, "retrieval score": e.retrieval_score, "NLI label": e.nli_label,
                       "NLI probability": e.nli_score, "mapped judgement": e.judgement} for e in resp.evidence])
            st.caption("Aggregation rule: a paper counts only if its NLI probability is at least 0.5. "
                       "SUPPORT and CONTRADICT papers together give CONFLICTING; only one kind gives that verdict; "
                       "otherwise UNKNOWN.")
        st.info(resp.caveat)
