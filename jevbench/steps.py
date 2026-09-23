"""Exploratory: does asking Jev in steps beat asking for the verdict directly?

Five Jev setups, each one request per post:

  monolithic   1 Choice: the verdict, as in the final benchmark (the control)
  yesno5       5 yes/no questions: rights, manner, other side, fair clash, spin
  severity2    2 Scores: how wrong was the poster, how wrong was the other side
  steps5       the two Scores, two Choices (what kind of act, what kind of
               conflict) and a spin Score, with no verdict question
  steps6       steps5 plus the direct verdict Choice

Step answers become four verdict probabilities through a small multinomial
logistic regression, scored out of fold (10-fold, stratified), so no post is
judged by a model fitted on it. The direct verdict is also refitted the same way,
so every setup gets the same recalibration and the comparison is fair.

This runs on a development pool (2023 posts, never sent to Jev before), not on
the 2025 holdout, so choosing a winner here cannot leak into the final results.

    python -m jevbench.steps sample                    # 100 posts -> data/steps-dev-100.jsonl
    python -m jevbench.steps sample --size 200 --out data/steps-dev-extra-200.jsonl \\
        --exclude data/steps-dev-100.jsonl             # 200 more, none repeated
    python -m jevbench.bench --data data/steps-dev-100.jsonl --models "~typesafe/jev-latest" \\
        --arms monolithic yesno5 severity2 steps5 steps6 --out runs/diagnostic/steps-dev-100.jsonl
    (the same for the extra 200, into runs/diagnostic/steps-dev-extra-200.jsonl)
    python -m jevbench.steps fit runs/diagnostic/steps-dev-*.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from .data import VERDICTS
from .questions import VERDICT_QUESTION
from .report import brier_one, stratified

LABELS = tuple(VERDICTS)
ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "final-clean-20260923.jsonl"
SAMPLE = ROOT / "data" / "steps-dev-100.jsonl"
PRIORS = ROOT / "data" / "final-ucb-2025.source.json"


def _noul(instructions: str, yes: str, no: str) -> dict:
    return {"type": "noul", "instructions": instructions, "criteria": {"true": yes, "false": no}}


def _choice(instructions: str, options: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": options}


def _score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def _severity(who: str) -> dict:
    return _score(f"How wrong was {who}'s conduct in this conflict?", [
        "Not wrong at all",
        "A minor lapse most people would forgive",
        f"Clearly wrong: {who} owes an apology",
        f"Seriously wrong: {who} caused real harm or unfairness",
    ])


VERDICT = _choice(VERDICT_QUESTION, dict(VERDICTS))

YESNO = {
    "op_within_rights": _noul(
        "Was the poster within their rights to want what they wanted?",
        "What the poster wanted was reasonable and theirs to decide",
        "What the poster wanted was unreasonable or not theirs to decide"),
    "op_harsh_manner": _noul(
        "Did the poster act harshly, rudely or excessively, whatever their reasons?",
        "The way the poster acted was harsh, rude or out of proportion",
        "The poster acted in a measured, considerate way"),
    "other_unreasonable": _noul(
        "Did another person in the story behave unreasonably, pushily or unfairly?",
        "Someone else demanded, pressured or acted unfairly",
        "The other people behaved reasonably"),
    "fair_clash": _noul(
        "Is this a clash between two reasonable needs, with nobody acting badly?",
        "Both sides want something reasonable and nobody behaved badly",
        "At least one side behaved badly"),
    "op_spin": _noul(
        "Is the poster leaving out or downplaying their own part?",
        "The account is one-sided or vague about the poster's own conduct",
        "The account reads as complete and even-handed"),
}

SEVERITY = {"op_severity": _severity("the poster"),
            "other_severity": _severity("the other party")}

KINDS = {
    "op_kind": _choice("Which best describes what the poster did?", {
        "reasonable_kind": "Wanted something reasonable and went about it considerately",
        "reasonable_harsh": "Wanted something reasonable but went about it harshly or rudely",
        "unreasonable": "Wanted something unreasonable or unfair to others",
        "bystander": "Mostly reacted to others; did nothing notable themselves",
    }),
    "conflict_kind": _choice("What kind of conflict is this?", {
        "one_side": "One side is clearly in the wrong",
        "both_bad": "Both sides behaved badly",
        "fair_clash": "Two reasonable needs clash and nobody behaved badly",
        "misunderstanding": "A misunderstanding or miscommunication more than a wrong",
    }),
    "op_spin": _score("How one-sided is the poster's account of their own part?", [
        "Even-handed", "Somewhat favorable to the poster", "Clearly spun in the poster's favor",
    ]),
}

ARMS = {
    "yesno5": lambda: dict(YESNO),
    "severity2": lambda: dict(SEVERITY),
    "steps5": lambda: {**SEVERITY, **KINDS},
    "steps6": lambda: {**SEVERITY, **KINDS, "verdict": VERDICT},
}


# --- Out-of-fold multinomial logistic regression, stdlib only --------------

def features(rec: dict) -> list[float]:
    """Every answer as numbers: a yes probability, or one probability per option."""
    out = [rec["probs"][k] for k in sorted(rec["probs"])]
    for k in sorted(rec["choices"]):
        p = rec["choices"][k]["probabilities"]
        out += [p[o] for o in sorted(p)]
    return out


def fit(x: list[list[float]], y: list[int], w: list[float],
        l2: float = 0.1, steps: int = 400, lr: float = 0.5) -> list[list[float]]:
    """Weighted softmax regression by gradient descent; x must be standardised.

    ponytail: fixed L2 and step count, no inner tuning; fine at 100 posts and
    under 20 features, tune by nested CV if a setup looks borderline.
    """
    d, k, total = len(x[0]) + 1, len(LABELS), sum(w)
    W = [[0.0] * d for _ in range(k)]
    rows = [xi + [1.0] for xi in x]
    for _ in range(steps):
        g = [[0.0] * d for _ in range(k)]
        for xi, yi, wi in zip(rows, y, w):
            p = softmax(W, xi)
            for c in range(k):
                err = wi * (p[c] - (c == yi)) / total
                for j in range(d):
                    g[c][j] += err * xi[j]
        for c in range(k):
            for j in range(d):
                W[c][j] -= lr * (g[c][j] + (l2 * W[c][j] if j < d - 1 else 0.0))
    return W


def softmax(W: list[list[float]], xi: list[float]) -> list[float]:
    z = [sum(a * b for a, b in zip(row, xi)) for row in W]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]


def out_of_fold(x: list[list[float]], y: list[int], w: list[float],
                folds: int = 10, seed: int = 0) -> list[dict[str, float]]:
    """Each post predicted by a model fitted on the other nine folds, split within class."""
    rng = random.Random(seed)
    fold = [0] * len(y)
    for c in set(y):
        idx = [i for i, yi in enumerate(y) if yi == c]
        rng.shuffle(idx)
        for n, i in enumerate(idx):
            fold[i] = n % folds
    preds: list[dict[str, float]] = [{}] * len(y)
    for f in range(folds):
        train = [i for i in range(len(y)) if fold[i] != f]
        cols = list(zip(*(x[i] for i in train)))
        mu = [sum(col) / len(col) for col in cols]
        sd = [math.sqrt(sum((v - m) ** 2 for v in col) / len(col)) or 1.0
              for col, m in zip(cols, mu)]
        z = lambda xi: [(v - m) / s for v, m, s in zip(xi, mu, sd)] + [1.0]
        W = fit([z(x[i])[:-1] for i in train], [y[i] for i in train], [w[i] for i in train])
        for i in range(len(y)):
            if fold[i] == f:
                preds[i] = dict(zip(LABELS, softmax(W, z(x[i]))))
    return preds


# --- Commands --------------------------------------------------------------

def sample(n_per: dict[str, int], out: Path, exclude: list[Path], seed: int = 20260924) -> None:
    """Draw from the pool, skipping posts already in `exclude`, so a sample can grow."""
    used = {json.loads(line)["id"] for p in exclude for line in p.read_text().splitlines() if line}
    rows = [r for line in POOL.read_text().splitlines() if line
            for r in [json.loads(line)] if r["id"] not in used]
    rng = random.Random(seed)
    chosen = []
    for v, n in n_per.items():
        bucket = [r for r in rows if r["verdict"] == v]
        chosen += rng.sample(bucket, n)
    rng.shuffle(chosen)
    if out.exists():
        raise SystemExit(f"{out} already exists")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen))
    print(f"wrote {out}: {dict(Counter(r['verdict'] for r in chosen))}")


def evaluate(logs: list[Path], balanced: bool = False) -> None:
    recs = [json.loads(line) for log in logs for line in log.read_text().splitlines() if line]
    recs = [r for r in recs if not r.get("error") and not r.get("parse_failed")]
    priors = json.loads(PRIORS.read_text())["eligible_mix"]
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_arm[r["arm"]].append(r)

    # Fit toward the population mix (74% NTA), the target the headline scores use.
    mix = Counter(r["verdict_true"] for r in next(iter(by_arm.values())))
    # --balanced instead weights the four verdicts equally, which asks whether
    # the steps help find the rare ESH and NAH posts at all.
    target = {v: 1.0 for v in LABELS} if balanced else priors
    pw = {v: target[v] / sum(target.values()) / (mix[v] / sum(mix.values())) for v in LABELS}
    if balanced:
        priors = target

    rows = []
    direct_raw = {r["item_id"]: r["choices"]["verdict"]["probabilities"] for r in by_arm["monolithic"]}
    setups = [("direct, raw (as in the benchmark)", "monolithic", None)] + [
        (name, arm, arm) for name, arm in [("direct, refitted", "monolithic"),
                                           ("yes/no ×5", "yesno5"), ("severity scores ×2", "severity2"),
                                           ("steps ×5 (no verdict)", "steps5"),
                                           ("steps ×5 + verdict", "steps6")] if arm in by_arm]
    ref: dict[str, float] = {}
    for name, arm, fitted in setups:
        rs = sorted(by_arm[arm], key=lambda r: r["item_id"])
        truth = [r["verdict_true"] for r in rs]
        if fitted:
            preds = out_of_fold([features(r) for r in rs], [LABELS.index(t) for t in truth],
                                [pw[t] for t in truth])
        else:
            preds = [{v: p / sum(direct_raw[r["item_id"]].values())
                      for v, p in direct_raw[r["item_id"]].items()} for r in rs]
        brier = {r["item_id"]: brier_one(p, t) for r, p, t in zip(rs, preds, truth)}
        if name == "direct, refitted":
            ref = brier

        def parts(values):
            out: dict[str, list[float]] = defaultdict(list)
            for r, t in zip(rs, truth):
                out[t].append(values[r["item_id"]])
            return out

        b, lo, hi = stratified(parts(brier), priors)
        hits = {r["item_id"]: float(max(p, key=p.get) == t) for r, p, t in zip(rs, preds, truth)}
        top1 = stratified(parts(hits), priors)[0]
        recall = {v: sum(hits[r["item_id"]] for r, t in zip(rs, truth) if t == v) / mix[v] for v in LABELS}
        delta = (stratified(parts({i: brier[i] - ref[i] for i in brier if i in ref}), priors)
                 if ref and fitted and name != "direct, refitted" else None)
        lat = sorted(r["latency_s"] for r in rs)[len(rs) // 2]
        cost = sum(r["billed_usd"] for r in rs) / len(rs) * 1000
        rows.append((name, rs[0]["n_questions"], b, lo, hi, delta, top1, recall, lat, cost))

    base = sum((priors[v] / sum(priors.values())) * brier_one(
        {u: priors[u] / sum(priors.values()) for u in LABELS}, v) for v in LABELS)
    print(f"{len(by_arm['monolithic'])} posts {dict(mix)} · weighted to "
          f"{'equal verdict shares' if balanced else 'the 2025 population mix'} · "
          f"no-model baseline Brier {base:.3f}\n")
    print("| Jev setup | Q | Weighted Brier ↓ [95% CI] | Δ vs direct refitted [95% CI] | "
          "Weighted top-1 | YTA / NTA / ESH / NAH recall | p50 | $/1k |")
    print("|---|---:|---:|---:|---:|---|---:|---:|")
    for name, q, b, lo, hi, delta, top1, rec, lat, cost in rows:
        d = f"{delta[0]:+.3f} [{delta[1]:+.3f}, {delta[2]:+.3f}]" if delta else ""
        r = " / ".join(f"{rec[v]:.0%}" for v in LABELS)
        print(f"| {name} | {q} | {b:.3f} [{lo:.3f}, {hi:.3f}] | {d} | {top1:.1%} | {r} | "
              f"{lat * 1000:.0f} ms | ${cost:.3f} |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("sample", help="draw development posts, 35/35/15/15 per 100")
    sm.add_argument("--size", type=int, default=100)
    sm.add_argument("--out", type=Path, default=SAMPLE)
    sm.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="earlier samples whose posts must not be drawn again")
    f = sub.add_parser("fit", help="score every setup in one or more run logs")
    f.add_argument("logs", type=Path, nargs="+")
    f.add_argument("--balanced", action="store_true",
                   help="weight the four verdicts equally instead of by population")
    args = ap.parse_args()
    if args.cmd == "sample":
        k = args.size // 100
        sample({"nta": 35 * k, "yta": 35 * k, "esh": 15 * k, "nah": 15 * k}, args.out, args.exclude)
    else:
        evaluate(args.logs, args.balanced)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
