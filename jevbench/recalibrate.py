"""Recalibration check, as planned in docs/RECALIBRATION_PLAN.md. No API calls.

Every configuration's saved four-verdict probabilities go through the same
cross-fitted logistic regression as Part 1, weighted to the eligible 2025
verdict mix; then Sonnet and Jev are compared again. A Jev-only check fits on
the 300 development posts and applies that fit to the final posts unchanged.

    python -m jevbench.recalibrate
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from .report import brier_one, by_arm, distribution, label, load, ordered, stratified
from .steps import LABELS, fit_predict, out_of_fold

ROOT = Path(__file__).resolve().parent.parent
PRIORS = ROOT / "data" / "final-ucb-2025.source.json"
SONNET = ("anthropic/claude-sonnet-5", "monolithic")
JEV = ("~typesafe/jev-latest", "monolithic")

Row = tuple[str, str, dict[str, float]]   # post id, true verdict, forecast


def usable(recs: list[dict]) -> list[Row]:
    """Posts with a valid four-verdict forecast, normalised to sum to 1."""
    return [(r["item_id"], r["verdict_true"], d) for r in recs
            if (d := distribution(r)) is not None]


def weights(rows: list[Row], priors: dict[str, float]) -> list[float]:
    """Per-post weights that make each verdict count at its population share."""
    mix = Counter(t for _, t, _ in rows)
    share = {v: priors[v] / sum(priors.values()) / (mix[v] / len(rows)) for v in LABELS}
    return [share[t] for _, t, _ in rows]


def cross_fitted(rows: list[Row], priors: dict[str, float]) -> list[Row]:
    """Each post's forecast recalibrated by a fit on the other nine folds."""
    x = [[d[v] for v in LABELS] for _, _, d in rows]
    y = [LABELS.index(t) for _, t, _ in rows]
    preds = out_of_fold(x, y, weights(rows, priors))
    return [(i, t, p) for (i, t, _), p in zip(rows, preds)]


def transferred(train: list[Row], test: list[Row], priors: dict[str, float]) -> list[Row]:
    """`test` recalibrated by a fit on `train` alone."""
    preds = fit_predict([[d[v] for v in LABELS] for _, _, d in train],
                        [LABELS.index(t) for _, t, _ in train], weights(train, priors),
                        [[d[v] for v in LABELS] for _, _, d in test])
    return [(i, t, p) for (i, t, _), p in zip(test, preds)]


def score(rows: list[Row], priors: dict[str, float]) -> tuple[float, float, float]:
    parts: dict[str, list[float]] = defaultdict(list)
    for _, t, p in rows:
        parts[t].append(brier_one(p, t))
    return stratified(parts, priors)


def paired(a: list[Row], b: list[Row], priors: dict[str, float]) -> tuple[float, float, float]:
    """Weighted Brier of `a` minus `b` on the posts both answered."""
    bb = {i: brier_one(p, t) for i, t, p in b}
    parts: dict[str, list[float]] = defaultdict(list)
    for i, t, p in a:
        if i in bb:
            parts[t].append(brier_one(p, t) - bb[i])
    return stratified(parts, priors)


def recall(rows: list[Row], verdict: str) -> float:
    hits = [max(p, key=p.get) == t for _, t, p in rows if t == verdict]
    return sum(hits) / len(hits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    priors = json.loads(PRIORS.read_text())["eligible_mix"]
    groups = by_arm(load(*map(Path, sorted(glob.glob(str(ROOT / "runs" / "final-*.jsonl"))))))
    raw = {key: usable(recs) for key, recs in ordered(groups)}
    recal = {key: cross_fitted(rows, priors) for key, rows in raw.items()}

    fmt = lambda s: f"{s[0]:.3f} [{s[1]:.3f}, {s[2]:.3f}]"
    dfmt = lambda s: f"{s[0]:+.3f} [{s[1]:+.3f}, {s[2]:+.3f}]"
    print("### Every configuration, recalibrated the same way\n")
    print("| Configuration | Posts | Raw weighted Brier ↓ | Recalibrated ↓ [95% CI] | "
          "Change [95% CI] | ESH / NAH recall, raw → recalibrated |")
    print("|---|---:|---:|---:|---:|---|")
    for key, rows in raw.items():
        r, c = score(rows, priors), score(recal[key], priors)
        rec = " / ".join(f"{recall(rows, v):.0%} → {recall(recal[key], v):.0%}" for v in ("esh", "nah"))
        print(f"| {label(*key)} | {len(rows)} | {r[0]:.3f} | {fmt(c)} | "
              f"{dfmt(paired(recal[key], rows, priors))} | {rec} |")

    print("\n### Primary: Sonnet minus Jev, weighted Brier\n")
    print("| | Difference [95% CI] |\n|---|---:|")
    print(f"| Raw (the headline) | {dfmt(paired(raw[SONNET], raw[JEV], priors))} |")
    print(f"| Both recalibrated | {dfmt(paired(recal[SONNET], recal[JEV], priors))} |")

    dev = usable([r for p in sorted(glob.glob(str(ROOT / "runs" / "diagnostic" / "steps-dev-*.jsonl")))
                  for line in Path(p).read_text().splitlines() if line
                  for r in [json.loads(line)] if r["arm"] == "monolithic"])
    moved = transferred(dev, raw[JEV], priors)
    print(f"\n### Secondary: Jev recalibrated on the {len(dev)} development posts only\n")
    print("| | Weighted Brier ↓ [95% CI] | ESH / NAH recall |\n|---|---:|---|")
    for name, rows in [("Jev raw", raw[JEV]), ("Jev, fit on development posts", moved),
                       ("Sonnet raw", raw[SONNET])]:
        print(f"| {name} | {fmt(score(rows, priors))} | "
              f"{recall(rows, 'esh'):.0%} / {recall(rows, 'nah'):.0%} |")
    print(f"\nSonnet raw minus Jev fit on development posts: "
          f"{dfmt(paired(raw[SONNET], moved, priors))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
