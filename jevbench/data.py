"""Sample r/AmItheAsshole posts from the HuggingFace datasets-server.

Each post carries the verdict Reddit reached (the top-voted judgment) plus the
two highest-scoring comments, so a demo can show what the crowd actually said
next to what the model said.

Reddit's verdict distribution is famously lopsided -- about 78% NTA in a random
sample -- so "always answer NTA" scores 78% without reading anything. We
stratify to make accuracy mean something, and report the base rate anyway.
"""

from __future__ import annotations

import json
import random
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "OsamaBsher/AITA-Reddit-Dataset"
SPLIT = "train"
SPLIT_ROWS = 270_709
PAGE = 100          # datasets-server caps length at 100

# The four canonical judgments. INFO ("not enough info") is excluded: it is a
# statement about the post rather than about the people in it, and the dataset
# labels it inconsistently.
VERDICTS = {
    "yta": "You're the asshole -- the poster is in the wrong",
    "nta": "Not the asshole -- the other party is in the wrong",
    "esh": "Everyone sucks here -- both sides behaved badly",
    "nah": "No assholes here -- a genuine conflict, nobody misbehaved",
}

# Target mix. Reddit's own is ~78/15/5/2, which would leave the interesting
# verdicts with almost no examples.
TARGET_MIX = {"nta": 0.35, "yta": 0.35, "nah": 0.15, "esh": 0.15}

# Posters often edit the verdict back into their own post ("EDIT: seems like
# I'm NTA, thanks all"). That hands the model the answer inside the state, so
# those posts are dropped before sampling rather than scored.
VERDICT_LEAK = re.compile(
    r"\b(?:yta|nta|esh|nah|you(?:\s+are|'re)\s+the\s+(?:asshole|a\s*hole)"
    r"|i(?:\s+(?:am|was)|'m)\s+(?:the|a(?:\s+\w+)?)\s+(?:asshole|a\s*hole)"
    r"|not\s+the\s+(?:asshole|a\s*hole)|everyone\s+sucks\s+here"
    r"|no\s+assholes?\s+here)\b", re.IGNORECASE,
)


def leaks_verdict(text: str) -> bool:
    return bool(VERDICT_LEAK.search(text))


@dataclass(frozen=True)
class Item:
    id: str
    title: str
    text: str
    verdict: str          # yta | nta | esh | nah
    score: int            # post karma, a rough proxy for how clear-cut it was
    top_comments: list[str]

    @property
    def body(self) -> str:
        """What the model sees: title and post, as a reader gets them."""
        return f"{self.title}\n\n{self.text}"


def _page(offset: int) -> list[dict]:
    q = urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": SPLIT,
         "offset": offset, "length": PAGE}
    )
    with urllib.request.urlopen(f"{ROWS_URL}?{q}", timeout=30) as r:
        return json.load(r)["rows"]


def sample(n: int = 200, seed: int = 0, min_chars: int = 400, max_chars: int = 4000,
           min_score: int = 5) -> list[Item]:
    """Stratified sample across the four verdicts.

    `min_score` drops posts the crowd barely engaged with: a verdict with three
    votes behind it is a noisier target than one with three hundred, and we are
    treating the crowd as ground truth.
    """
    rng = random.Random(seed)
    want = {v: max(1, round(n * frac)) for v, frac in TARGET_MIX.items()}
    got: dict[str, list[Item]] = {v: [] for v in VERDICTS}
    seen_pages: set[int] = set()
    seen_ids: set[str] = set()

    while any(len(got[v]) < want[v] for v in want):
        offset = rng.randrange(0, SPLIT_ROWS - PAGE)
        if offset in seen_pages:
            continue
        seen_pages.add(offset)
        if len(seen_pages) > 400:
            have = {v: len(got[v]) for v in got}
            raise RuntimeError(f"Could not fill sample after {len(seen_pages)} pages: {have}")

        for row in _page(offset):
            r = row["row"]
            v = (r.get("verdict") or "").strip().lower()
            text = (r.get("text") or "").strip()
            if v not in VERDICTS or r["id"] in seen_ids:
                continue
            if not (min_chars <= len(text) <= max_chars) or (r.get("score") or 0) < min_score:
                continue
            if leaks_verdict(text) or leaks_verdict(r.get("title") or ""):
                continue
            if len(got[v]) >= want[v]:
                continue
            seen_ids.add(r["id"])
            got[v].append(Item(
                id=str(r["id"]),
                title=(r.get("title") or "").strip(),
                text=text,
                verdict=v,
                score=int(r.get("score") or 0),
                top_comments=[c.strip() for c in (r.get("comment1"), r.get("comment2")) if c],
            ))

    items = [i for v in got for i in got[v][:want[v]]]
    rng.shuffle(items)
    return items


def load_or_fetch(path: Path, n: int = 200, seed: int = 0) -> list[Item]:
    """Cache the sample so repeated runs judge the identical posts.

    Refetches when the cache holds fewer posts than asked for -- otherwise a
    cached 40-post sample would silently satisfy `-n 200` and the run would be
    a fifth of the requested size with nothing saying so.
    """
    if path.exists():
        cached = [Item(**json.loads(line)) for line in path.read_text().splitlines() if line]
        if len(cached) >= n:
            return cached[:n]
    items = sample(n, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(i)) + "\n" for i in items))
    return items
