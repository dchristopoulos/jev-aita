"""Score saved answers against how commenters split, not just the official flair.

Exploratory, added after the main results: same 770 posts, same saved answers,
no new model calls. The target is each post's share of NTA/YTA/ESH/NAH comments
(INFO dropped, the rest rescaled to 1). Distances are class-weighted like the
main table; the confidence correlation is a plain Pearson r over posts.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .data import VERDICTS
from .five_question_followup import DEV, DIRECT, evaluate as followup
from .report import by_arm, distribution, label, load, ordered, stratified, _by_class

ROOT = Path(__file__).resolve().parent.parent
FIVE = ROOT / "runs/followup-yesno5-2025.jsonl"
BASE = "No model: base rates"
FLAIR = "Official verdict, taken as 100% certain"


def crowd_split(row: dict, prefix: str = "comments_prop_") -> dict[str, float] | None:
    shares = {v: row[prefix + v.upper()] for v in VERDICTS}
    total = sum(shares.values())
    return {v: s / total for v, s in shares.items()} if total else None


def distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Brier against a soft target: 0 when the model matches the crowd exactly."""
    return sum((p[v] - q[v]) ** 2 for v in VERDICTS)


def evaluate(recs: list[dict], source: list[dict], priors: dict[str, float],
             fixed: dict[str, dict[str, dict[str, float]]] | None = None) -> str:
    raw = {str(r["id"]): r for r in source}
    total = sum(priors.values())
    base = {v: priors[v] / total for v in VERDICTS}
    forecasts: dict[str, dict[str, dict[str, float]]] = {}
    truth: dict[str, str] = {}
    notes = []
    for (model, arm), group in ordered(by_arm(recs)):
        name = label(model, arm)
        forecasts[name] = {r["item_id"]: p for r in group if (p := distribution(r))}
        truth.update({r["item_id"]: r["verdict_true"] for r in group})
        if len(forecasts[name]) < len(group):
            notes.append(f"{name}: {len(group) - len(forecasts[name])} malformed answers left out")
    # Adjusted Jev rows: the 2023-trained regressions from the five-question follow-up.
    forecasts.update(fixed or {})
    forecasts[BASE] = {i: base for i in truth}
    forecasts[FLAIR] = {i: {v: float(v == t) for v in VERDICTS} for i, t in truth.items()}

    one_vote = {i: crowd_split(raw[i]) for i in truth}
    upvoted = {i: crowd_split(raw[i], "comments_prop_weighted_") for i in truth}
    ids = [i for i in truth if one_vote[i] and upvoted[i]]
    if len(ids) < len(truth):
        notes.append(f"{len(truth) - len(ids)} posts without verdict comments left out")

    def losses(name: str, target: dict) -> dict[str, float]:
        return {i: distance(forecasts[name][i], target[i]) for i in ids if i in forecasts[name]}

    def weighted(values: dict[str, float]) -> tuple[float, float, float]:
        return stratified(_by_class([(truth[i], v) for i, v in values.items()]), priors)

    rows = []
    for name, preds in forecasts.items():
        mean, lo, hi = weighted(losses(name, one_vote))
        up = weighted(losses(name, upvoted))[0]
        r = ""
        if name not in (BASE, FLAIR):  # undefined for a constant forecast
            try:
                r = format(statistics.correlation(
                    [max(preds[i].values()) for i in ids if i in preds],
                    [max(one_vote[i].values()) for i in ids if i in preds]), "+.2f")
            except statistics.StatisticsError:  # e.g. every thread unanimous
                r = "n/a"
        rows.append((mean, f"| {name} | {mean:.3f} [{lo:.3f}, {hi:.3f}] | {up:.3f} | {r} |"))

    def paired(a: str, b: str) -> str:
        la, lb = losses(a, one_vote), losses(b, one_vote)
        d, lo, hi = weighted({i: la[i] - lb[i] for i in la if i in lb})
        return f"{a} minus {b}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    names = [n for n in forecasts if n not in (BASE, FLAIR)]
    sonnet = next(n for n in names if n.startswith("Sonnet"))
    jev = next(n for n in names if n.startswith("Jev · direct"))
    diffs = [paired(sonnet, jev)] + [paired(n, BASE) for n in names]
    return ("| Model | Distance from comment split ↓ [95% CI] | From upvote-weighted split ↓ "
            "| Confidence vs. crowd agreement (r) |\n|---|---:|---:|---:|\n"
            + "\n".join(row for _, row in sorted(rows)) + "\n\nPaired differences "
            "(negative = closer to the commenters):\n" + "\n".join(diffs)
            + ("\n\n" + "; ".join(notes) if notes else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-raw", type=Path, default=ROOT / "data/ucb-2025-raw.jsonl")
    ap.add_argument("--priors", type=Path, default=ROOT / "data/final-ucb-2025.source.json")
    args = ap.parse_args()
    priors = json.loads(args.priors.read_text())["eligible_mix"]
    fixed: dict = {}
    followup(load(*DEV), load(DIRECT), load(FIVE), priors, fixed_out=fixed)
    fixed = {f"Jev · {name} (adjusted)": preds for name, preds in fixed.items()}
    source = [json.loads(l) for l in args.source_raw.read_text().splitlines() if l]
    recs = load(*sorted((ROOT / "runs").glob("final-*-2025.jsonl")))
    print(evaluate(recs, source, priors, fixed))


if __name__ == "__main__":
    main()
