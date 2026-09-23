"""Walk through one post: what Jev was asked, what it answered, what that scores as.

Shows both arms: the direct four-way question, and the two yes/no questions
combined into four verdicts. Answers come from the final log, or from fresh
calls with --live.

    python -m jevbench.show                   # first post in the sample
    python -m jevbench.show 1hvkncw           # a specific post id
    python -m jevbench.show 1hvkncw --live    # ask Jev now (2 calls, ~$0.0001)
    python -m jevbench.show --random          # a random real post from Hugging Face (live)
    python -m jevbench.show --text "AITA for ..."   # judge your own post (live)
    pbpaste | python -m jevbench.show --text -     # ...or pipe it in
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import shutil
import sys
import textwrap

from .bench import DATA, RUNS
from .client import ask
from .data import VERDICTS, Item, leaks_verdict, load
from .questions import balanced, balanced_distribution, monolithic
from .sample_2025 import FLAIR, HF_ROWS, get_json

LOG = RUNS / "final-jev-2025.jsonl"
COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
WIDTH = min(shutil.get_terminal_size().columns, 100)


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def rule(title: str, right: str = "") -> str:
    fill = WIDTH - len(title) - len(right) - 6
    return "\n" + c(f"━━ {title} " + "━" * max(fill, 2) + (f" {right}" if right else ""), "1;36")


def wrap(text: str, indent: str = "  ") -> list[str]:
    return [line for para in text.splitlines()
            for line in (textwrap.wrap(para, WIDTH - len(indent)) or [""])]


def bars(probs: dict[str, float], truth: str) -> None:
    top = max(probs, key=probs.get)
    for v, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        bar = "█" * round(p * 40) or "▏"
        line = f"  {v.upper():3}  {p:5.2f}  {c(bar, '32' if v == truth else '2')}"
        tags = (["picked"] if v == top else []) + (["forum verdict"] if v == truth else [])
        print(line + (c("  ← " + " · ".join(tags), "1") if tags else ""))


def verdict_line(probs: dict[str, float], truth: str | None) -> None:
    top = max(probs, key=probs.get)
    if truth is None:
        print(f"\n  Jev picked {c(top.upper(), '1')}: {VERDICTS[top]}")
        return
    ok = top == truth
    print(f"\n  Jev picked {c(top.upper(), '1')} · forum said {truth.upper()} · "
          + c("✓ right" if ok else "✗ wrong", "1;32" if ok else "1;31")
          + f" · Brier {sum((p - (v == truth)) ** 2 for v, p in probs.items()):.3f}")


def fetch_random(rng: random.Random) -> Item:
    """A real post with Reddit's verdict, from the same D-Lab dataset, via the
    Hugging Face row API. Same filters as the benchmark sample."""
    total = get_json(f"{HF_ROWS}&offset=0&length=1")["num_rows_total"]
    for _ in range(5):
        rows = get_json(f"{HF_ROWS}&offset={rng.randrange(max(total - 100, 1))}&length=100")["rows"]
        ok = [r for r in (x["row"] for x in rows)
              if FLAIR.get(r.get("link_flair_text")) and r.get("title")
              and 400 <= len((r.get("selftext_cleaned") or "").strip()) <= 4000
              and not leaks_verdict(r["title"]) and not leaks_verdict(r["selftext_cleaned"])]
        if ok:
            r = rng.choice(ok)
            return Item(str(r["id"]), r["title"].strip(), r["selftext_cleaned"].strip(),
                        FLAIR[r["link_flair_text"]], int(r.get("score") or 0), [])
    raise OSError("no usable post in five pages")


def meta(r: dict) -> str:
    return f"{r['latency_s'] * 1000:.0f} ms · ${r['billed_usd']:.6f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id", nargs="?", help="post id (default: the first post in the sample)")
    ap.add_argument("--live", action="store_true", help="call Jev now instead of reading the final log")
    ap.add_argument("--short", action="store_true", help="print only the first lines of the post")
    ap.add_argument("--random", action="store_true",
                    help="fetch a random real post with Reddit's verdict from Hugging Face; "
                         "always a live call")
    ap.add_argument("--text", help="judge this post instead (title on the first line; '-' reads stdin); "
                                   "always a live call")
    args = ap.parse_args()

    if args.random:
        try:
            item = fetch_random(random.Random())
        except OSError as e:
            ap.error(f"could not fetch a post from Hugging Face: {e}")
        args.live = True
    elif args.text is not None:
        text = (sys.stdin.read() if args.text == "-" else args.text).strip()
        if not text:
            ap.error("--text is empty")
        title, _, body = text.partition("\n")
        item = Item("your-post", title.strip(), body.strip(), "", 0, [])
        args.live = True
    elif not DATA.exists():
        ap.error(f"{DATA} is missing; rebuild it with the steps under 'Reproduce' in the README")
    else:
        items = load(DATA)
        item = next((i for i in items if i.id == args.id), None) if args.id else items[0]
        if item is None:
            ap.error(f"no post {args.id!r} in {DATA.name}")
    truth = item.verdict or None

    arms = {"monolithic": monolithic(), "balanced": balanced()}
    if args.live:
        runs = {}
        for arm, qs in arms.items():
            a = ask(item.body, qs, "~typesafe/jev-latest")
            runs[arm] = {"probs": a.probs, "latency_s": a.latency_s, "billed_usd": a.billed_usd,
                         "resolved_model": a.resolved_model,
                         "choices": {k: dataclasses.asdict(v) for k, v in a.choices.items()}}
        source = "live call"
    else:
        runs = {r["arm"]: r for line in LOG.read_text().splitlines() if line
                for r in [json.loads(line)] if r["item_id"] == item.id}
        if set(arms) - set(runs):
            ap.error(f"{LOG.name} is missing an answer for {item.id}")
        source = LOG.name
    model = runs["monolithic"]["resolved_model"]

    # The post, as Jev sees it (title + body is the whole `state`).
    print(rule(f"POST {item.id}", f"{len(item.body)} chars"))
    print(c("  " + item.title, "1"))
    body = wrap(item.text)
    shown = body[:10] if args.short and len(body) > 12 else body
    print("\n".join(c("  " + line, "2") for line in shown))
    if len(shown) < len(body):
        print(c(f"  … {len(body) - len(shown)} more lines (drop --short to show)", "2;3"))
    if truth:
        print(f"\n  Forum verdict: {c(truth.upper(), '1')} ({VERDICTS[truth]})")
    print(f"{'' if truth else chr(10)}  Model: {model} · answers from {source}")

    # Arm 1: one Choice question over the four verdicts.
    r, q = runs["monolithic"], arms["monolithic"]["verdict"]
    print(rule("STEP 1 · Direct question", meta(r)))
    print(c("  Asked:", "1"), q["instructions"])
    for code, meaning in q["criteria"].items():
        print(c(f"    {code.upper():3}  {meaning}", "2"))
    ans = r["choices"]["verdict"]
    print(c("\n  Answered:", "1") + f" choice {ans['choice'].upper()}, confidence {ans['confidence']:.2f}\n")
    bars(ans["probabilities"], truth)
    verdict_line(ans["probabilities"], truth)

    # Arm 2: two independent yes/no questions, multiplied into four verdicts.
    r = runs["balanced"]
    print(rule("STEP 2 · Two yes/no questions", meta(r)))
    for n, (qid, qq) in enumerate(arms["balanced"].items(), 1):
        p = r["probs"][qid]
        print(c(f"  Q{n}", "1") + f"  {qq['instructions']}")
        print(f"      yes {c(f'{p:.2f}', '1')}  {c('█' * round(p * 30), '36')}{c('░' * (30 - round(p * 30)), '2')}  no {1 - p:.2f}")
    a, b = r["probs"]["poster_at_fault"], r["probs"]["other_at_fault"]
    print(c("\n  Combined", "1") + " (treats Q1 and Q2 as independent):")
    dist = balanced_distribution(r["probs"])
    for v, form, x, y in [("yta", "Q1 × (1−Q2)", a, 1 - b), ("nta", "(1−Q1) × Q2", 1 - a, b),
                          ("esh", "Q1 × Q2", a, b), ("nah", "(1−Q1) × (1−Q2)", 1 - a, 1 - b)]:
        print(c(f"    {v.upper()}  {form:15} = {x:.2f} × {y:.2f} = ", "2") + f"{dist[v]:.2f}")
    print()
    bars(dist, truth)
    verdict_line(dist, truth)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
