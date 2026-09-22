"""Calibration against a probability, not a coin flip.

The AITA benchmark scores a verdict against Reddit's flair: one binary label
decided by a crowd vote. That tests accuracy, but it cannot really test
*calibration* -- when Jev says 0.7, there is no ground truth saying how often
0.7 should be right, only a yes/no that happens to disagree 30% of the time.

Civil Comments carries the thing AITA lacks: `toxicity` is the FRACTION of
human raters who marked a comment toxic. A 0.6 means 6 of 10 people said yes.
That fraction *is* the probability a randomly drawn rater says yes, and it is
exactly what a calibrated model should output. So here the target and the
prediction live on the same scale and can be compared directly.

Run:  python -m jevbench.civil -n 300
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .client import ApiError, FatalApiError, ask
from .metrics import brier, ece, percentile, reliability_bins

HF = "https://datasets-server.huggingface.co/rows"
DATASET = "google/civil_comments"
CACHE = Path("data/civil.jsonl")

# Bins chosen so the disagreement band is over-sampled. A random draw is ~71%
# zeros, and a model can score a great ECE on that by answering 0.0 forever.
# The interesting question is what it does when humans themselves split.
BANDS = [(0.0, 0.001), (0.001, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]

TOXIC = {
    "is_toxic": {
        "type": "noul",
        "instructions": (
            "Would a typical reader consider this comment toxic -- rude, "
            "disrespectful, or likely to make someone leave the conversation?"
        ),
        "criteria": {
            "true": "A reader would call this toxic",
            "false": "A reader would not call this toxic",
        },
    }
}


@dataclass(frozen=True)
class Comment:
    text: str
    toxicity: float   # fraction of human raters who said toxic


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "jevbench"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch(per_band: int, seed: int = 0, max_pages: int = 400) -> list[Comment]:
    """Stratified sample across rater-agreement bands.

    Pages at random offsets rather than reading from row 0, so the sample is
    not just the oldest comments in the dump.
    """
    rng = random.Random(seed)
    q = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": "train"})
    buckets: list[list[Comment]] = [[] for _ in BANDS]
    seen: set[str] = set()

    for _ in range(max_pages):
        if all(len(b) >= per_band for b in buckets):
            break
        off = rng.randrange(0, 1_800_000)
        for row in _get(f"{HF}?{q}&offset={off}&length=100")["rows"]:
            d = row["row"]
            text = (d.get("text") or "").strip()
            tox = float(d.get("toxicity") or 0.0)
            if not (20 <= len(text) <= 1500) or text in seen:
                continue
            for i, (lo, hi) in enumerate(BANDS):
                if lo <= tox < hi and len(buckets[i]) < per_band:
                    buckets[i].append(Comment(text, tox))
                    seen.add(text)
                    break

    out = [c for b in buckets for c in b]
    rng.shuffle(out)     # so a budget stop does not truncate one whole band
    return out


def load_or_fetch(n: int, seed: int = 0) -> list[Comment]:
    per_band = max(1, n // len(BANDS))
    want = per_band * len(BANDS)
    if CACHE.exists():
        rows = [json.loads(l) for l in CACHE.open()]
        if len(rows) >= want:
            return [Comment(r["text"], r["toxicity"]) for r in rows[:want]]
    items = fetch(per_band, seed)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("w") as fh:
        for c in items:
            fh.write(json.dumps({"text": c.text, "toxicity": c.toxicity}) + "\n")
    return items


def run(models: list[str], items: list[Comment], out: Path, budget: float = 0.0) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    started, spend, n_ok, n_err = time.time(), 0.0, 0, 0
    with out.open("w") as fh:
        for model in models:
            print(f"\n{model}  [{len(items)} comments]", file=sys.stderr)
            for i, c in enumerate(items, 1):
                try:
                    a = ask(c.text, TOXIC, model)
                    p = a.probs.get("is_toxic")
                    rec = {
                        "ts": time.time(), "model": model, "text_chars": len(c.text),
                        "toxicity_true": c.toxicity, "predicted": p,
                        "latency_s": round(a.latency_s, 4), "cost_usd": a.cost_usd,
                        "prompt_tokens": a.prompt_tokens,
                        "completion_tokens": a.completion_tokens, "error": None,
                    }
                    spend += a.cost_usd
                    n_ok += 1
                except FatalApiError:
                    raise
                except ApiError as e:
                    rec = {"ts": time.time(), "model": model, "toxicity_true": c.toxicity,
                           "predicted": None, "cost_usd": 0.0, "error": str(e)}
                    n_err += 1
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if i % 25 == 0 or i == len(items):
                    print(f"  [{i}/{len(items)}]  ${spend:.4f}", file=sys.stderr)
                if budget and spend >= budget:
                    print(f"\n  budget stop: ${spend:.4f} >= ${budget:.2f}", file=sys.stderr)
                    return {"ok": n_ok, "errors": n_err, "spend_usd": round(spend, 6),
                            "wall_s": round(time.time() - started, 1), "stopped_early": True}
    return {"ok": n_ok, "errors": n_err, "spend_usd": round(spend, 6),
            "wall_s": round(time.time() - started, 1), "stopped_early": False}


def report(paths: list[Path]) -> str:
    recs = [json.loads(l) for p in paths for l in p.open()]
    good = [r for r in recs if r.get("predicted") is not None]
    lines = ["## Calibration against human rater fractions", "",
             f"{len(good)} comments from `{DATASET}`. Target is the fraction of "
             "human raters who called the comment toxic, so a perfectly "
             "calibrated model's mean prediction equals it in every bin.", "",
             "| Model | n | ECE ↓ | Brier ↓ | mean pred | mean truth | p50 latency |",
             "|---|---|---|---|---|---|---|"]
    for model in sorted({r["model"] for r in good}):
        sub = [r for r in good if r["model"] == model]
        preds = [r["predicted"] for r in sub]
        truth = [r["toxicity_true"] for r in sub]
        lat = [r["latency_s"] for r in sub if r.get("latency_s")]
        p50 = f"{percentile(lat, 0.5) * 1000:.0f} ms" if lat else "n/a"
        lines.append(
            f"| `{model}` | {len(sub)} | {ece(preds, truth):.3f} | "
            f"{brier(preds, truth):.3f} | {sum(preds) / len(preds):.3f} | "
            f"{sum(truth) / len(truth):.3f} | {p50} |"
        )

    for model in sorted({r["model"] for r in good}):
        sub = [r for r in good if r["model"] == model]
        bins = reliability_bins([r["predicted"] for r in sub],
                                [r["toxicity_true"] for r in sub])
        lines += ["", f"### `{model}` reliability", "",
                  "| predicted | n | mean pred | mean truth | gap |", "|---|---|---|---|---|"]
        for b in bins:
            lines.append(f"| {b.lo:.1f}–{b.hi:.1f} | {b.n} | {b.mean_pred:.3f} | "
                         f"{b.mean_truth:.3f} | {b.mean_truth - b.mean_pred:+.3f} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=["~typesafe/jev-latest"])
    ap.add_argument("--budget", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=Path("runs/civil.jsonl"))
    ap.add_argument("--report", nargs="*", type=Path,
                    help="skip the run; render these logs instead")
    a = ap.parse_args()

    if a.report is not None:
        print(report(a.report or [a.out]))
        return 0

    items = load_or_fetch(a.n, a.seed)
    print(f"{len(items)} comments, mean toxicity {sum(c.toxicity for c in items)/len(items):.3f}",
          file=sys.stderr)
    totals = run(a.models, items, a.out, a.budget)
    print(f"\n{totals['ok']} ok, {totals['errors']} failed · "
          f"${totals['spend_usd']:.4f} · {totals['wall_s']:.0f}s", file=sys.stderr)
    a.out.with_suffix(".meta.json").write_text(json.dumps(
        {"dataset": DATASET, "n": len(items), "seed": a.seed,
         "models": a.models, **totals}, indent=2))
    print(report([a.out]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
