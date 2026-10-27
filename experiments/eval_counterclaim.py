"""Does the counter-claim consistency check catch wrong verdicts?

    python -m experiments.eval_counterclaim --nli-model models/nli-scifact
    python -m experiments.eval_counterclaim --nli-model facebook/bart-large-mnli --limit 50

For every dev claim: verify it, build its opposite (verification/counterclaim.py), verify that
too, and label the pair consistent / inconsistent / inconclusive. Then compare the verdict's
error rate inside each group. If "inconsistent" verdicts are wrong much more often than
"consistent" ones, the flag is useful: a user can trust consistent verdicts more and treat
flagged ones as unreliable. Also reports the accuracy the system would have if it abstained
on every flagged claim.

Writes experiments/results/counterclaim_eval_<model>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from data.loader import load_claims, load_config
from verification.counterclaim import build_counterclaim, judge_consistency
from verification.factory import build_verifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-model", required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    data_cfg = load_config("configs/data.yaml")
    dev = load_claims(Path(data_cfg["paths"]["raw_dir"]), data_cfg["paths"]["claims_dev_file"], has_labels=True)
    if args.limit:
        dev = dev[: args.limit]
    verifier = build_verifier(args.nli_model)

    rows, t0 = [], time.perf_counter()
    for i, c in enumerate(dev, start=1):
        r = verifier.verify(c.claim)
        cc = build_counterclaim(c.claim)
        r2 = verifier.verify(cc.text)
        status, _ = judge_consistency(r.verdict, r2.verdict)
        rows.append({"claim_id": c.id, "claim": c.claim, "counter": cc.text, "method": cc.method,
                     "gold": c.gold_verdict().value, "verdict": r.verdict, "counter_verdict": r2.verdict,
                     "status": status, "correct": r.verdict == c.gold_verdict().value})
        if i % 10 == 0 or i == len(dev):
            el = time.perf_counter() - t0
            print(f"  [{i}/{len(dev)}] {el:.0f}s elapsed, eta {el / i * (len(dev) - i):.0f}s", flush=True)

    groups = defaultdict(list)
    for r in rows:
        groups[r["status"]].append(r["correct"])
    by_status = {s: {"claims": len(v), "wrong": int(len(v) - sum(v)),
                     "error_rate": round(1 - sum(v) / len(v), 4)} for s, v in groups.items()}
    kept = [r["correct"] for r in rows if r["status"] != "inconsistent"]
    decided = [r for r in rows if r["verdict"] in ("SUPPORTED", "CONTRADICTED")]
    decided_kept = [r["correct"] for r in decided if r["status"] != "inconsistent"]
    report = {
        "nli_model": args.nli_model, "num_claims": len(rows),
        "overall_accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
        "by_status": by_status,
        "by_method": dict(Counter(r["method"] for r in rows)),
        "counter_verdict_counts": dict(Counter(r["counter_verdict"] for r in rows)),
        "abstain_on_inconsistent": {
            "claims_kept": len(kept), "accuracy_of_kept": round(sum(kept) / len(kept), 4) if kept else None},
        "decided_claims_only": {
            "claims": len(decided),
            "accuracy": round(sum(r["correct"] for r in decided) / len(decided), 4) if decided else None,
            "accuracy_if_abstain_on_inconsistent": round(sum(decided_kept) / len(decided_kept), 4) if decided_kept else None,
            "kept": len(decided_kept)},
        "total_time_seconds": round(time.perf_counter() - t0, 1),
        "_provenance": "Produced by experiments/eval_counterclaim.py on the user's machine.",
    }
    tag = args.nli_model.rstrip("/").split("/")[-1] + (f"_limit{args.limit}" if args.limit else "")
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"counterclaim_eval_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / f"counterclaim_eval_{tag}_predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
