import json
from pathlib import Path

from jevbench.crowd import crowd_split, evaluate
from jevbench.report import load

ROOT = Path(__file__).resolve().parent.parent


def test_split_drops_info_and_rescales():
    row = {"comments_prop_NTA": 0.4, "comments_prop_YTA": 0.2, "comments_prop_ESH": 0.2,
           "comments_prop_NAH": 0.0, "comments_prop_INFO": 0.2}
    assert crowd_split(row) == {"yta": 0.25, "nta": 0.5, "esh": 0.25, "nah": 0.0}


def test_official_verdict_scores_zero_when_every_commenter_agrees():
    recs = load(ROOT / "runs/final-jev-2025.jsonl", ROOT / "runs/final-sonnet-2025.jsonl")
    # Synthetic comments: every commenter gave the official verdict.
    source = [{"id": r["item_id"], **{f"comments_prop{w}_{v}": float(v.lower() == r["verdict_true"])
                                      for w in ("", "_weighted") for v in ("NTA", "YTA", "ESH", "NAH")}}
              for r in recs if r["model"].startswith("anthropic")]
    priors = json.loads((ROOT / "data/final-ucb-2025.source.json").read_text())["eligible_mix"]
    table = evaluate(recs, source, priors)
    assert "| Official verdict, taken as 100% certain | 0.000 [0.000, 0.000] | 0.000 |  |" in table
    assert "Sonnet 5 · direct minus Jev · direct:" in table
