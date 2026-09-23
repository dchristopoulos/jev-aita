"""Score the pre-specified five-question follow-up on the 2025 posts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .data import VERDICTS
from .metrics import percentile
from .report import brier_one, distribution, load, predicted, stratified
from .steps import YESNO, features, fit_predict

ROOT = Path(__file__).resolve().parent.parent
DEV = (ROOT / "runs/diagnostic/steps-dev-100.jsonl",
       ROOT / "runs/diagnostic/steps-dev-extra-200.jsonl")
DIRECT = ROOT / "runs/final-jev-2025.jsonl"
PRIORS = ROOT / "data/final-ucb-2025.source.json"


def arm_rows(recs: list[dict], arm: str, count: int) -> dict[str, dict]:
    rows = [r for r in recs if r["arm"] == arm]
    if len(rows) != count or len({r["item_id"] for r in rows}) != count:
        raise ValueError(f"{arm}: expected {count} distinct posts, got {len(rows)} rows")
    if any(r.get("error") or r.get("parse_failed") for r in rows):
        raise ValueError(f"{arm}: failed or malformed answers; run is incomplete")
    if arm == "yesno5" and any(
        set(r["probs"]) != set(YESNO) or
        any(not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
            for p in r["probs"].values()) for r in rows
    ):
        raise ValueError("yesno5: missing or invalid probabilities")
    return {r["item_id"]: r for r in rows}


def evaluate(dev: list[dict], direct: list[dict], five: list[dict], priors: dict) -> str:
    dev_one = arm_rows(dev, "monolithic", 300)
    dev_five = arm_rows(dev, "yesno5", 300)
    raw = arm_rows(direct, "monolithic", 770)
    new = arm_rows(five, "yesno5", 770)
    if set(dev_one) != set(dev_five) or set(raw) != set(new):
        raise ValueError("paired arms judged different posts")
    if set(dev_one) & set(raw):
        raise ValueError("development and evaluation posts overlap")
    for groups in ((dev_one, dev_five), (raw, new)):
        if any(groups[0][i]["verdict_true"] != groups[1][i]["verdict_true"] for i in groups[0]):
            raise ValueError("paired arms disagree on labels")

    ids = [r["item_id"] for r in direct if r["arm"] == "monolithic"]
    dev_ids = sorted(dev_one)
    mix = Counter(dev_one[i]["verdict_true"] for i in dev_ids)
    total = sum(priors.values())
    weights = [priors[dev_one[i]["verdict_true"]] / total /
               (mix[dev_one[i]["verdict_true"]] / len(dev_ids)) for i in dev_ids]

    def fitted(train: dict[str, dict], test: dict[str, dict]) -> dict[str, dict[str, float]]:
        predictions = fit_predict([features(train[i]) for i in dev_ids],
                                  [list(VERDICTS).index(dev_one[i]["verdict_true"]) for i in dev_ids],
                                  weights, [features(test[i]) for i in ids])
        return dict(zip(ids, predictions))

    preds = {"Direct, fitted on 2023": fitted(dev_one, raw),
             "Five yes/no, fitted on 2023": fitted(dev_five, new)}
    raw_probs = {i: distribution(raw[i]) for i in ids}
    if any(p is None for p in raw_probs.values()):
        raise ValueError("raw direct probabilities are incomplete")
    preds = {"Direct, raw": raw_probs, **preds}
    records = {"Direct, raw": raw, "Direct, fitted on 2023": raw,
               "Five yes/no, fitted on 2023": new}
    truth = {i: raw[i]["verdict_true"] for i in ids}

    def by_class(values: dict[str, float]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = defaultdict(list)
        for i in ids:
            out[truth[i]].append(values[i])
        return out

    losses = {}
    lines = ["| Jev setup | Weighted Brier ↓ [95% CI] | Weighted top-1 [95% CI] | "
             "Macro recall | Median call | $ per 1,000 posts |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, p in preds.items():
        losses[name] = {i: brier_one(p[i], truth[i]) for i in ids}
        b, blo, bhi = stratified(by_class(losses[name]), priors)
        hits = {i: float((predicted(raw[i])[0] if name == "Direct, raw" else
                          max(p[i], key=p[i].get)) == truth[i]) for i in ids}
        a, alo, ahi = stratified(by_class(hits), priors)
        macro = sum(sum(hits[i] for i in ids if truth[i] == v) /
                    sum(truth[i] == v for i in ids) for v in VERDICTS) / len(VERDICTS)
        rs = records[name]
        latency = percentile([r["latency_s"] for r in rs.values()], 0.5)
        cost = sum(r["billed_usd"] for r in rs.values()) / len(rs) * 1000
        lines.append(f"| {name} | {b:.3f} [{blo:.3f}, {bhi:.3f}] | "
                     f"{a:.1%} [{alo:.1%}, {ahi:.1%}] | {macro:.1%} | "
                     f"{latency:.2f} s | ${cost:.3f} |")
        if name == "Five yes/no, fitted on 2023":
            lines.append("Rare-class recall: " + ", ".join(
                f"{v.upper()} {sum(hits[i] for i in ids if truth[i] == v) / sum(truth[i] == v for i in ids):.1%}"
                for v in ("esh", "nah")))

    diff = {i: losses["Five yes/no, fitted on 2023"][i] -
            losses["Direct, fitted on 2023"][i] for i in ids}
    d, lo, hi = stratified(by_class(diff), priors)
    decision = "better" if hi < 0 else "worse" if lo > 0 else "inconclusive"
    lines.append(f"\nPaired weighted Brier difference, five minus direct fitted: "
                 f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]; {decision}.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("five_log", type=Path)
    args = ap.parse_args()
    print(evaluate(load(*DEV), load(DIRECT), load(args.five_log),
                   json.loads(PRIORS.read_text())["eligible_mix"]))


if __name__ == "__main__":
    main()
