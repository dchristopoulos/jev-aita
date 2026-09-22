"""Freeze a reproducible 2025 AITA holdout from the Berkeley D-Lab dilemmas."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .data import Item, leaks_verdict

DATASET = "ucberkeley-dlab/fragility-moral-judgment-llms"
REVISION = "cb4c298cbfa93ce9cdf56685f12a55b0a6928110"
FLAIR = {"Not the A-hole": "nta", "Asshole": "yta",
         "Everyone Sucks": "esh", "No A-holes here": "nah"}
TARGET = {"nta": 365, "yta": 285, "esh": 65, "nah": 55}


def select(raw: list[dict], seed: int = 20260923) -> tuple[list[Item], list[dict], dict]:
    buckets: dict[str, list[dict]] = {v: [] for v in TARGET}
    for row in raw:
        verdict = FLAIR.get(row.get("link_flair_text"))
        body = (row.get("selftext_cleaned") or "").strip()
        title = (row.get("title") or "").strip()
        if (verdict and 400 <= len(body) <= 4000 and title and
                not leaks_verdict(title) and not leaks_verdict(body)):
            buckets[verdict].append(row)
    eligible = {v: len(rows) for v, rows in buckets.items()}
    if any(eligible[v] < n for v, n in TARGET.items()):
        raise ValueError(f"not enough eligible posts: {eligible}")

    rng = random.Random(seed)
    chosen = []
    for verdict, count in TARGET.items():
        rng.shuffle(buckets[verdict])
        chosen += [(row, verdict) for row in buckets[verdict][:count]]
    rng.shuffle(chosen)

    items = [Item(str(r["id"]), r["title"].strip(), r["selftext_cleaned"].strip(),
                  verdict, int(r.get("score") or 0), []) for r, verdict in chosen]
    labels = [{"id": str(r["id"]), "created_utc": r["created_utc"],
               "n_verdicts": r["n_verdicts"],
               "comment_proportions": {v.lower(): r[f"comments_prop_{v}"]
                                       for v in ("NTA", "YTA", "ESH", "NAH", "INFO")}}
              for r, _ in chosen]
    assert len({i.id for i in items}) == len(items)
    return items, labels, eligible


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args()
    labels_path = args.out.with_suffix(".labels.jsonl")
    meta_path = args.out.with_suffix(".source.json")
    if args.out.exists() or labels_path.exists():
        ap.error("output already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = args.raw.read_bytes()
    items, labels, eligible = select([json.loads(l) for l in raw_bytes.splitlines()], args.seed)
    meta = json.dumps({"dataset": DATASET, "revision": REVISION,
                       "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                       "seed": args.seed, "eligible_mix": eligible,
                       "sample_mix": dict(Counter(i.verdict for i in items)),
                       "criteria": "official four-class flair; 400-4000 cleaned body characters; no verdict phrase; no karma cutoff"},
                      indent=2) + "\n"
    # The committed source record doubles as the check on a rebuild.
    if meta_path.exists() and meta_path.read_text() != meta:
        ap.error(f"{meta_path} does not match this source export and seed")
    args.out.write_text("".join(json.dumps(asdict(i), ensure_ascii=False) + "\n" for i in items))
    labels_path.write_text("".join(json.dumps(r) + "\n" for r in labels))
    meta_path.write_text(meta)
    print(f"{len(items)} frozen posts: {dict(Counter(i.verdict for i in items))}; "
          f"eligible population: {eligible}")


if __name__ == "__main__":
    main()
