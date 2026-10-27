"""Command-line demo.

    python -m experiments.demo "Monoclonal antibody targeting of N-cadherin inhibits metastasis."
    python -m experiments.demo                      # interactive: type claims, empty line to quit

The first claim takes longer (loads the corpus and the NLI model); later ones are faster.
"""

from __future__ import annotations

import argparse
import textwrap

from verification.factory import build_verifier


def show(resp) -> None:
    print("\n" + "=" * 78)
    print(f"CLAIM:      {resp.claim}")
    print(f"VERDICT:    {resp.verdict}      (model confidence {resp.confidence:.2f}, not calibrated)")
    print(f"WHY:        {resp.explanation}")
    for i, e in enumerate(resp.evidence, start=1):
        print(f"\n  [Paper {i}] {e.title}")
        print(f"     retrieval score {e.retrieval_score} | judge says {e.judgement} "
              f"({e.nli_label}, p={e.nli_score})")
        print(textwrap.fill("Most claim-relevant sentences: " + e.passage, 76,
                            initial_indent="     ", subsequent_indent="     "))
    print(f"\n  time: {resp.elapsed_seconds}s")
    print("  " + textwrap.fill(resp.caveat, 74, subsequent_indent="  "))
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("claim", nargs="?")
    ap.add_argument("--nli-model", help="override the model in configs/verification.yaml (demo.nli_model)")
    args = ap.parse_args()

    print("Loading corpus and model (first time can take a minute)...")
    verifier = build_verifier(args.nli_model)
    if args.claim:
        show(verifier.verify(args.claim))
        return
    while True:
        claim = input("\nClaim (empty line to quit): ").strip()
        if not claim:
            break
        show(verifier.verify(claim))


if __name__ == "__main__":
    main()
