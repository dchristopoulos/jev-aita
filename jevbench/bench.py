"""Run the benchmark, recording one JSONL line per API request plus a manifest.

A record is per *request*, not per question, because latency and cost are
properties of the request -- which is exactly what the fan-out arms test.

Two things are deliberate:

Failures are recorded, not just printed. A run where Sonnet times out on 20% of
posts is a different result from a clean run, and a log that silently omits them
looks identical to one that never hit them.

The manifest pins everything needed to interpret the numbers later: seed,
dataset, price list, git commit, wall time, totals. Prices change and models
move behind `-latest`; a run log without them is uninterpretable in six months.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .client import PRICING, ApiError, FatalApiError, ask
from .data import DATASET, Item, load_or_fetch
from .questions import decomposed, full, monolithic

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sample.jsonl"
RUNS = ROOT / "runs"

ARMS = {"monolithic": monolithic, "decomposed": decomposed, "full": full}
# Cheap by default. Jev bills $0.042/M in and nothing out; gpt-5-nano is the
# baseline that makes the cost claim awkward, which is the point of having it.
# A frontier model costs ~50x the whole rest of the run -- opt in with --models.
DEFAULT_MODELS = ["~typesafe/jev-latest", "openai/gpt-5-nano"]

BUDGET_STOP = "budget reached"


def _git_sha() -> str:
    """Which version of the questions produced these numbers."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _record(model: str, arm: str, item: Item, a, questions: dict) -> dict:
    return {
        "ts": time.time(),
        "model": model,
        "resolved_model": a.resolved_model,
        "request_id": a.request_id,
        "arm": arm,
        "item_id": item.id,
        "verdict_true": item.verdict,
        "post_score": item.score,
        "state_chars": len(item.body),
        "n_questions": a.n_questions,
        "probs": {k: round(v, 5) for k, v in a.probs.items()},
        "choices": {k: {"choice": v.choice,
                        "confidence": round(v.confidence, 5),
                        "probabilities": {o: round(p, 5) for o, p in v.probabilities.items()}}
                    for k, v in a.choices.items()},
        "scores": {k: {"position": round(v.position, 5),
                       "raw_score": round(v.raw_score, 5),
                       "confidence": round(v.confidence, 5)}
                   for k, v in a.scores.items()},
        "latency_s": round(a.latency_s, 4),
        "attempts": a.attempts,
        "prompt_tokens": a.prompt_tokens,
        "completion_tokens": a.completion_tokens,
        "total_tokens": a.total_tokens,
        "cost_usd": a.cost_usd,
        "billed_usd": a.billed_usd,
        "provider": str(a.raw.get("provider", "")),
        "parse_failed": a.parse_failed,
        "error": None,
    }


def _error_record(model: str, arm: str, item: Item, n_questions: int, err: str) -> dict:
    """A failed request still happened, and the report must be able to see it."""
    return {
        "ts": time.time(), "model": model, "resolved_model": "", "request_id": "",
        "arm": arm, "item_id": item.id, "verdict_true": item.verdict,
        "post_score": item.score, "state_chars": len(item.body),
        "n_questions": n_questions, "probs": {}, "choices": {}, "scores": {},
        "latency_s": 0.0, "attempts": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "total_tokens": 0, "cost_usd": 0.0, "billed_usd": 0.0, "provider": "",
        "parse_failed": False, "error": err[:300],
    }


def estimate(models: list[str], arms: list[str], items: list[Item]) -> float:
    """Rough upper bound on what a run will cost, before spending anything.

    Tokens are estimated at 4 characters each plus question overhead. It is a
    guess, so it rounds against us: better to over-warn than to over-spend.
    """
    chars = sum(len(i.body) for i in items)
    total = 0.0
    for model in models:
        in_m, out_m = PRICING.get(model, (0.0, 0.0))
        for arm in arms:
            nq = len(ARMS[arm]())
            prompt = chars / 4 + len(items) * nq * 40     # questions repeat per request
            completion = 0 if model.startswith("~typesafe/") else len(items) * nq * 12
            total += (prompt * in_m + completion * out_m) / 1_000_000
    return total * 1.25   # headroom for the estimate being wrong


def run(models: list[str], arms: list[str], items: list[Item], out: Path,
        budget: float = 0.0) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    n_ok = n_err = 0
    spend = 0.0
    stopped = ""

    with out.open("w") as fh:
        for model in models:
            for arm in arms:
                questions = ARMS[arm]()
                print(f"\n{model}  [{arm}: {len(questions)}q]", file=sys.stderr)
                for i, item in enumerate(items, 1):
                    try:
                        a = ask(item.body, questions, model)
                        rec = _record(model, arm, item, a, questions)
                        spend += a.cost_usd
                        n_ok += 1
                    except FatalApiError:
                        # No key, bad key, no credit: stop now rather than
                        # print the same message once per remaining post.
                        raise
                    except ApiError as e:
                        rec = _error_record(model, arm, item, len(questions), str(e))
                        n_err += 1
                        print(f"  [{i}/{len(items)}] error: {e}", file=sys.stderr)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()   # a crash at post 180 keeps the first 179
                    if i % 25 == 0 or i == len(items):
                        print(f"  [{i}/{len(items)}]  ${spend:.4f} so far", file=sys.stderr)

                    if budget and spend >= budget:
                        # Stop the moment the cap is hit. Whatever was measured
                        # is already on disk and the report handles a part-run.
                        stopped = (f"{BUDGET_STOP}: ${spend:.4f} >= ${budget:.2f} "
                                   f"during {model}/{arm} at post {i}/{len(items)}")
                        print(f"\n  {stopped}", file=sys.stderr)
                        return {"ok": n_ok, "errors": n_err, "spend_usd": round(spend, 6),
                                "wall_s": round(time.time() - started, 1),
                                "stopped_early": stopped}

    return {"ok": n_ok, "errors": n_err, "spend_usd": round(spend, 6),
            "wall_s": round(time.time() - started, 1), "stopped_early": stopped}


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark Jev against LLM baselines.")
    ap.add_argument("-n", type=int, default=200, help="posts to sample (stratified)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--budget", type=float, default=1.00,
                    help="hard cap in USD; the run stops the moment it is reached "
                         "(0 disables)")
    ap.add_argument("--estimate", action="store_true",
                    help="print the projected cost and exit without spending")
    args = ap.parse_args()

    items = load_or_fetch(DATA, args.n, args.seed)
    mix = Counter(i.verdict for i in items)
    print(f"{len(items)} posts  {dict(mix)}", file=sys.stderr)

    projected = estimate(args.models, args.arms, items)
    n_req = len(items) * len(args.models) * len(args.arms)
    print(f"{n_req} requests · projected ~${projected:.3f}"
          + (f" · cap ${args.budget:.2f}" if args.budget else " · no cap"), file=sys.stderr)
    if args.estimate:
        return 0
    if args.budget and projected > args.budget:
        print("Projected cost exceeds the cap. Raise --budget or lower -n.", file=sys.stderr)
        return 1

    out = args.out or RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    try:
        totals = run(args.models, args.arms, items, out, args.budget)
    except ApiError as e:
        print(f"\nfatal: {e}", file=sys.stderr)
        return 1

    meta = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _git_sha(),
        "dataset": DATASET,
        "n_posts": len(items),
        "seed": args.seed,
        "verdict_mix": dict(mix),
        "models": args.models,
        "arms": args.arms,
        "questions_per_arm": {a: len(ARMS[a]()) for a in args.arms},
        "pricing_usd_per_mtok": {m: PRICING.get(m) for m in args.models},
        "budget_usd": args.budget,
        "projected_usd": round(projected, 6),
        **totals,
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\n{totals['ok']} ok, {totals['errors']} failed · "
          f"${totals['spend_usd']:.4f} · {totals['wall_s']:.0f}s", file=sys.stderr)
    print(f"wrote {out} and {meta_path}\n"
          f"now: python -m jevbench.report {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
