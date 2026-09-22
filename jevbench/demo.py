"""Judge a handful of real posts and show the working.

The point of this file is that you can read the post yourself and disagree.
A benchmark table tells you a model scored 61%; this tells you *which* posts it
got wrong and what it was thinking, which is the only way to tell a model that
is badly calibrated from a crowd that is.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from .client import ApiError, ask
from .data import VERDICTS, Item, load_or_fetch
from .questions import compose_verdict, full

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sample.jsonl"

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, RED, YELLOW, CYAN = "\033[32m", "\033[31m", "\033[33m", "\033[36m"


def _wrap(text: str, width: int, indent: str = "  ") -> str:
    out = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        out += textwrap.wrap(para, width=width, initial_indent=indent,
                             subsequent_indent=indent)
    return "\n".join(out)


def _bar(p: float, width: int = 18) -> str:
    filled = round(p * width)
    return "█" * filled + "·" * (width - filled)


def show_post(item: Item, width: int = 88, full_text: bool = False) -> None:
    """Print a post and what Reddit said. No API key, no model call."""
    print(f"\n{BOLD}{'─' * width}{RESET}")
    print(f"{BOLD}{item.title[:width-2]}{RESET}")
    print(f"{DIM}  r/AmItheAsshole · {item.score} points · post {item.id}{RESET}\n")

    body = item.text if full_text else item.text[:900] + ("…" if len(item.text) > 900 else "")
    print(_wrap(body, width - 4))

    print(f"\n{BOLD}  Reddit's verdict{RESET}  "
          f"{CYAN}{item.verdict.upper()}{RESET}  {DIM}{VERDICTS[item.verdict]}{RESET}")
    for i, c in enumerate(item.top_comments, 1):
        print(f"\n{DIM}  top comment {i}{RESET}")
        print(f"{DIM}{_wrap(c[:500], width - 4, '    ')}{RESET}")
    print()


def show(item: Item, model: str, width: int = 88, full_text: bool = False) -> bool:
    """Judge one post and print the comparison. Returns True if the model agreed."""
    questions = full()
    a = ask(item.body, questions, model)

    print(f"\n{BOLD}{'─' * width}{RESET}")
    print(f"{BOLD}{item.title[:width-2]}{RESET}")
    print(f"{DIM}  r/AmItheAsshole · {item.score} points · post {item.id}{RESET}\n")

    body = item.text if full_text else item.text[:900] + ("…" if len(item.text) > 900 else "")
    print(_wrap(body, width - 4))

    direct = a.choices["verdict"]
    composed = compose_verdict(a.probs)
    truth = item.verdict

    print(f"\n{BOLD}  The model's reasoning{RESET}  {DIM}(8 yes/no questions, one call){RESET}")
    for qid, p in sorted(a.probs.items(), key=lambda kv: -kv[1]):
        color = RED if p >= 0.5 else DIM
        print(f"    {color}{_bar(p)}{RESET} {p:4.0%}  {qid}")

    if "severity" in a.scores:
        sev = a.scores["severity"]
        print(f"\n    severity {sev.position:.0%} {DIM}(confidence {sev.confidence:.0%}){RESET}")

    def tag(v: str) -> str:
        c = GREEN if v == truth else RED
        return f"{c}{v.upper()}{RESET}"

    print(f"\n{BOLD}  Verdicts{RESET}")
    print(f"    Reddit said           {BOLD}{CYAN}{truth.upper()}{RESET}  {DIM}{VERDICTS[truth]}{RESET}")
    print(f"    Asked directly        {tag(direct.choice)}  "
          f"{DIM}confidence {direct.confidence:.0%}{RESET}")
    print(f"    Worked out in code    {tag(composed)}  "
          f"{DIM}from the 8 answers above{RESET}")

    if direct.probabilities:
        dist = "  ".join(f"{k.upper()} {v:.0%}" for k, v in
                         sorted(direct.probabilities.items(), key=lambda kv: -kv[1]))
        print(f"    {DIM}distribution: {dist}{RESET}")

    if item.top_comments:
        print(f"\n{BOLD}  Top comment on Reddit{RESET}")
        print(f"{DIM}{_wrap(item.top_comments[0][:400], width - 4, '    ')}{RESET}")

    agreed = direct.choice == truth
    mark = f"{GREEN}agrees with Reddit{RESET}" if agreed else f"{YELLOW}disagrees with Reddit{RESET}"
    print(f"\n  → {mark}")
    return agreed


def main() -> int:
    ap = argparse.ArgumentParser(description="Judge real AITA posts and show the working.")
    ap.add_argument("-n", type=int, default=3, help="posts to judge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="~typesafe/jev-latest")
    ap.add_argument("--verdict", choices=list(VERDICTS), help="only posts with this verdict")
    ap.add_argument("--full-text", action="store_true", help="do not truncate the post")
    ap.add_argument("--read", action="store_true",
                    help="just read the posts and Reddit's verdicts -- no model, no API key")
    args = ap.parse_args()

    items = load_or_fetch(DATA, max(args.n * 8, 40), args.seed)
    if args.verdict:
        items = [i for i in items if i.verdict == args.verdict]
    if not items:
        print("No posts match that filter.", file=sys.stderr)
        return 1

    if args.read:
        for item in items[: args.n]:
            show_post(item, full_text=args.full_text)
        return 0

    agreed = 0
    for item in items[: args.n]:
        try:
            agreed += show(item, args.model, full_text=args.full_text)
        except ApiError as e:
            print(f"\nerror on post {item.id}: {e}", file=sys.stderr)
            return 1

    n = min(args.n, len(items))
    print(f"\n{BOLD}{'─' * 88}{RESET}")
    print(f"  Agreed with Reddit on {agreed}/{n}.")
    print(f"  {DIM}Disagreements are the interesting ones -- read the post and pick a side.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
