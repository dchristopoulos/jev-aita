"""Freeze a reproducible 2025 AITA holdout from the Berkeley D-Lab dilemmas.

`--fetch` downloads the dilemmas through the Hugging Face row API first. The
API serves the dataset's current version, so the download is checked against
`raw_sha256` in the committed source record and refused if it has drifted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .data import Item, leaks_verdict

DATASET = "ucberkeley-dlab/fragility-moral-judgment-llms"
REVISION = "cb4c298cbfa93ce9cdf56685f12a55b0a6928110"
FLAIR = {"Not the A-hole": "nta", "Asshole": "yta",
         "Everyone Sucks": "esh", "No A-holes here": "nah"}
TARGET = {"nta": 365, "yta": 285, "esh": 65, "nah": 55}
HF_ROWS = ("https://datasets-server.huggingface.co/rows"
           f"?dataset={DATASET}&config=dilemmas&split=train")
SOURCE = Path(__file__).resolve().parent.parent / "data" / "final-ucb-2025.source.json"


def get_json(url: str, tries: int = 6) -> dict:
    """GET with backoff: the row API rate-limits a full 30-page download."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
            time.sleep(int(e.headers.get("Retry-After") or 0) or 2 ** (attempt + 1))
    raise AssertionError("unreachable")


def fetch_raw(out: Path) -> None:
    """Every dilemma row, as one JSON line each, in the bytes the final runs used."""
    rows: list[dict] = []
    while True:
        page = get_json(f"{HF_ROWS}&offset={len(rows)}&length=100")
        rows += [r["row"] for r in page["rows"]]
        if not page["rows"] or len(rows) >= page["num_rows_total"]:
            break
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
    got = hashlib.sha256(data).hexdigest()
    want = json.loads(SOURCE.read_text())["raw_sha256"]
    if got != want:
        raise SystemExit(f"Hugging Face returned {len(rows)} rows hashing to {got[:12]}, "
                         f"not the pinned {want[:12]}; the dataset has changed since the runs")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"{len(rows)} rows -> {out}, sha256 matches the pinned source")


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
    ap.add_argument("--fetch", action="store_true",
                    help="download --raw from Hugging Face first (checked against its pinned hash)")
    args = ap.parse_args()
    if args.fetch:
        fetch_raw(args.raw)
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
