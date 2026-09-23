"""Run the benchmark, recording one JSONL line per API request plus a manifest.

A record is per *request*, not per question, because latency and cost are
properties of the request -- which is exactly what the fan-out arms test.

Two things are deliberate:

Failures are recorded, not just printed. A run where Sonnet times out on 20% of
posts is a different result from a clean run, and a log that silently omits them
looks identical to one that never hit them.

The manifest pins everything needed to interpret the numbers later: sample
hash, git commit and whether the tree was dirty, the exact prompt text, price
list, wall time, totals. Prices change and models
move behind `-latest`; a run log without them is uninterpretable in six months.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .client import (LOCAL_PREFIX, PRICING, ApiError, FatalApiError, production_verdict_prompt,
                     standard_choice_prompt, ask, local_chat_url)
from .data import Item, load
from .questions import balanced, monolithic
from .sample_2025 import DATASET, REVISION
from .steps import ARMS as STEP_ARMS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "final-ucb-2025.jsonl"
RUNS = ROOT / "runs"

# standard_choice is the chat-only single-label check; see docs/FINAL_PROTOCOL.md.
ARMS = {"monolithic": monolithic, "balanced": balanced, "standard_choice": monolithic,
        **STEP_ARMS}
JEV_ONLY = {"balanced", *STEP_ARMS}
# Cheap by default. Jev bills $0.042/M in and nothing out; gpt-5-nano is the
# baseline that makes the cost claim awkward, which is the point of having it.
# A frontier model costs ~50x the whole rest of the run -- opt in with --models.
DEFAULT_MODELS = ["~typesafe/jev-latest", "openai/gpt-5-nano"]

BUDGET_STOP = "budget reached"


def _host() -> dict:
    """The machine a local model ran on. macOS only; empty elsewhere."""
    def sysctl(key: str) -> str:
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    mem = sysctl("hw.memsize")
    return {"cpu": sysctl("machdep.cpu.brand_string"),
            "ram_gb": round(int(mem) / 2**30) if mem.isdigit() else None,
            "model": sysctl("hw.model")}


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _prompts(models: list[str], arms: list[str]) -> dict:
    """The exact text each arm sent, so a log can be checked without the code."""
    out = {}
    for arm in arms:
        questions = ARMS[arm]()
        if any(m.startswith("~typesafe/") for m in models):
            out[f"jev/{arm}"] = questions
        if arm not in JEV_ONLY and any(not m.startswith("~typesafe/") for m in models):
            prompt = standard_choice_prompt if arm == "standard_choice" else production_verdict_prompt
            out[f"chat/{arm}"] = prompt(questions["verdict"])
    return out


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
        "latency_s": round(a.latency_s, 4),
        "attempts": a.attempts,
        "prompt_tokens": a.prompt_tokens,
        "completion_tokens": a.completion_tokens,
        "total_tokens": a.total_tokens,
        # Thinking a model did before answering. Billed as output but never
        # shown, so it is the hidden part of both latency and cost.
        "reasoning_tokens": ((a.raw.get("usage") or {})
                             .get("completion_tokens_details") or {}).get("reasoning_tokens"),
        # Reported only by LM Studio's /api/v0 endpoint, null elsewhere.
        "tokens_per_second": (a.raw.get("stats") or {}).get("tokens_per_second"),
        "time_to_first_token_s": (a.raw.get("stats") or {}).get("time_to_first_token"),
        "model_info": a.raw.get("model_info"),
        "runtime": (a.raw.get("runtime") or {}).get("name"),
        "cost_usd": a.cost_usd,
        "billed_usd": a.billed_usd,
        "provider": str(a.raw.get("provider", "")),
        "parse_failed": a.parse_failed,
        # What the model actually returned, before parsing, so a parser change
        # or a malformed answer can be checked later: Jev's typed answers, or
        # the chat reply text.
        "raw_output": (a.raw.get("answers") if "answers" in a.raw else
                       ((a.raw.get("choices") or [{}])[0].get("message") or {}).get("content")),
        "error": None,
    }


def _error_record(model: str, arm: str, item: Item, n_questions: int, err: str) -> dict:
    """A failed request still happened, and the report must be able to see it."""
    return {
        "ts": time.time(), "model": model, "resolved_model": "", "request_id": "",
        "arm": arm, "item_id": item.id, "verdict_true": item.verdict,
        "post_score": item.score, "state_chars": len(item.body),
        "n_questions": n_questions, "probs": {}, "choices": {},
        "latency_s": 0.0, "attempts": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "total_tokens": 0, "cost_usd": 0.0, "billed_usd": 0.0, "provider": "",
        "parse_failed": False, "error": err[:300],
    }


def estimate(models: list[str], arms: list[str], items: list[Item]) -> float:
    """Rough guess at what a run will cost, before spending anything.

    Tokens are estimated at 4 characters each plus question overhead, with 25%
    headroom. It ignores reasoning tokens and the chat system prompt, so it has
    underestimated real runs; the --budget cap is what actually limits spend.
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
            if model.startswith(LOCAL_PREFIX) and items:
                # The first request makes the server load the model into memory,
                # which can take a minute. Timed like the rest it would land in
                # the latency percentiles, so it is sent once, untimed, unlogged.
                t0 = time.time()
                ask("A short warmup request.", ARMS[arms[0]](), model,
                    label_only=arms[0] == "standard_choice")
                print(f"\n{model}  loaded and warm in {time.time() - t0:.1f}s", file=sys.stderr)
            for arm in arms:
                questions = ARMS[arm]()
                print(f"\n{model}  [{arm}: {len(questions)}q]", file=sys.stderr)
                for i, item in enumerate(items, 1):
                    try:
                        a = ask(item.body, questions, model,
                                label_only=arm == "standard_choice")
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
    ap.add_argument("--data", type=Path, default=DATA,
                    help="frozen sample from jevbench.sample_2025 (default: the final sample)")
    ap.add_argument("-n", type=int, default=None,
                    help="judge only the first N posts, e.g. for a cheap smoke run")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--arms", nargs="+", default=["monolithic"], choices=list(ARMS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--budget", type=float, default=1.00,
                    help="hard cap in USD; the run stops the moment it is reached "
                         "(0 disables)")
    ap.add_argument("--estimate", action="store_true",
                    help="print the projected cost and exit without spending")
    args = ap.parse_args()

    if "standard_choice" in args.arms and any(m.startswith("~typesafe/") for m in args.models):
        ap.error("standard_choice is a chat-only prompt")
    if JEV_ONLY & set(args.arms) and any(not m.startswith("~typesafe/") for m in args.models):
        ap.error(f"{', '.join(sorted(JEV_ONLY & set(args.arms)))}: Jev-only arms")
    if not args.data.exists():
        ap.error(f"{args.data} not found; build it with python -m jevbench.sample_2025")
    items = load(args.data, args.n)
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
    if out.exists() or out.with_suffix(".meta.json").exists():
        print(f"Refusing to overwrite {out}", file=sys.stderr)
        return 1
    sample_sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
    code_sha = hashlib.sha256(b"".join(
        p.read_bytes() for p in sorted((ROOT / "jevbench").glob("*.py")))).hexdigest()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        totals = run(args.models, args.arms, items, out, args.budget)
    except ApiError as e:
        print(f"\nfatal: {e}", file=sys.stderr)
        return 1

    meta = {
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        # Uncommitted changes mean git_sha alone does not identify the code.
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "sample_path": str(args.data),
        "sample_sha256": sample_sha,
        "code_sha256": code_sha,
        "n_posts": len(items),
        "verdict_mix": dict(mix),
        "models": args.models,
        # Only meaningful for local models, where the hardware is the result.
        **({"local_server": local_chat_url(), "host": _host()}
           if any(m.startswith(LOCAL_PREFIX) for m in args.models) else {}),
        "arms": args.arms,
        "questions_per_arm": {a: len(ARMS[a]()) for a in args.arms},
        "prompts": _prompts(args.models, args.arms),
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
