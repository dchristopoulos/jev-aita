"""Score saved answers against how commenters split, not just the official flair.

Exploratory, added after the main results: same 770 posts, same saved answers,
no new model calls. The target is each post's share of NTA/YTA/ESH/NAH comments
(INFO dropped, the rest rescaled to 1).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .data import VERDICTS
from .report import by_arm, distribution, label, load, ordered, stratified, _by_class

ROOT = Path(__file__).resolve().parent.parent


def crowd_split(row: dict, prefix: str) -> dict[str, float] | None:
    shares = {v: row[prefix + v.upper()] for v in VERDICTS}
    total = sum(shares.values())
    return {v: s / total for v, s in shares.items()} if total else None


def distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Brier against a soft target: 0 when the model matches the crowd exactly."""
    return sum((p[v] - q[v]) ** 2 for v in VERDICTS)


def evaluate(recs: list[dict], source: list[dict], priors: dict[str, float]) -> str:
    raw = {str(r["id"]): r for r in source}
    groups = by_arm(recs)
    total = sum(priors.values())
    base = {v: priors[v] / total for v in VERDICTS}
    rows, notes = [], []

    def row(name: str, pairs: list[tuple[str, dict, dict, dict]]) -> None:
        # pairs: (flair, model probs, one-vote split, upvote split)
        mean, lo, hi = stratified(_by_class([(t, distance(p, q)) for t, p, q, _ in pairs]), priors)
        up = stratified(_by_class([(t, distance(p, w)) for t, p, _, w in pairs]), priors)[0]
        r = statistics.correlation([max(p.values()) for _, p, _, _ in pairs],
                                   [max(q.values()) for _, _, q, _ in pairs])
        rows.append(f"| {name} | {mean:.3f} [{lo:.3f}, {hi:.3f}] | {up:.3f} | {r:+.2f} |")

    for (model, arm), group in ordered(groups):
        pairs = []
        for rec in group:
            p = distribution(rec)
            q = crowd_split(raw[rec["item_id"]], "comments_prop_")
            w = crowd_split(raw[rec["item_id"]], "comments_prop_weighted_")
            if p and q and w:
                pairs.append((rec["verdict_true"], p, q, w))
        if len(pairs) < len(group):
            notes.append(f"{label(model, arm)}: {len(group) - len(pairs)} posts left out")
        row(label(model, arm), pairs)

    ref = next(iter(groups.values()))
    splits = [(r["verdict_true"], raw[r["item_id"]]) for r in ref]
    splits = [(t, crowd_split(s, "comments_prop_"), crowd_split(s, "comments_prop_weighted_"))
              for t, s in splits]
    splits = [(t, q, w) for t, q, w in splits if q and w]
    # Reference rows. Correlation is undefined for a constant forecast.
    for name, make in (("*No model: base rates*", lambda t: base),
                       ("*Official flair at 100%*", lambda t: {v: float(v == t) for v in VERDICTS})):
        mean, lo, hi = stratified(_by_class([(t, distance(make(t), q)) for t, q, _ in splits]), priors)
        up = stratified(_by_class([(t, distance(make(t), w)) for t, _, w in splits]), priors)[0]
        rows.append(f"| {name} | {mean:.3f} [{lo:.3f}, {hi:.3f}] | {up:.3f} | |")

    return ("| Model | Distance from comment split ↓ [95% CI] | From upvote-weighted split ↓ "
            "| Confidence vs. crowd agreement (r) |\n|---|---:|---:|---:|\n"
            + "\n".join(rows) + ("\n\n" + "; ".join(notes) if notes else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-raw", type=Path, default=ROOT / "data/ucb-2025-raw.jsonl")
    ap.add_argument("--priors", type=Path, default=ROOT / "data/final-ucb-2025.source.json")
    args = ap.parse_args()
    recs = load(*sorted((ROOT / "runs").glob("final-*-2025.jsonl")))
    source = [json.loads(l) for l in args.source_raw.read_text().splitlines() if l]
    print(evaluate(recs, source, json.loads(args.priors.read_text())["eligible_mix"]))


if __name__ == "__main__":
    main()
