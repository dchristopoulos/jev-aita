"""End-to-end smoke test for the report.

Catches the class of bug where a section function exists but is unreachable
from main() -- every table passes its own unit test and the CLI still crashes.
Uses a hand-built run log, so no API key or network is needed.
"""

import json

import pytest

from jevbench.questions import DIMENSIONS
from jevbench.report import (
    baseline_accuracy, by_arm, confidence_gate_table, confusion_table,
    fanout_note, headline_table, load, predicted, reliability_svg, severity_table,
)


def _record(arm, verdict, guess, conf=0.9, model="~typesafe/jev-latest"):
    nouls = ["op_broke_agreement"] if arm == "monolithic" else list(DIMENSIONS)
    # Make the composed verdict land on `guess` for the noul-only arm.
    probs = {d: 0.0 for d in nouls}
    if guess in ("yta", "esh"):
        probs["op_broke_agreement"] = 0.9
    if guess in ("nta", "esh") and "other_behaved_badly" in probs:
        probs["other_behaved_badly"] = 0.9
    rec = {
        "model": model, "arm": arm, "item_id": "1",
        "n_questions": len(nouls) + (2 if arm == "full" else 0),
        "probs": probs, "choices": {}, "scores": {},
        "verdict_true": verdict, "post_score": 100, "latency_s": 0.2,
        "prompt_tokens": 500, "completion_tokens": 0, "cost_usd": 2.1e-5,
        "parse_failed": False,
    }
    if arm in ("monolithic", "full"):
        rec["choices"] = {"verdict": {"choice": guess, "confidence": conf,
                                      "probabilities": {guess: conf}}}
    if arm == "full":
        rec["scores"] = {"severity": {"position": 0.5, "raw_score": 2.5,
                                      "confidence": conf}}
    return rec


@pytest.fixture
def run_log(tmp_path):
    recs = []
    for arm in ("monolithic", "decomposed", "full"):
        for verdict, guess in (("yta", "yta"), ("nta", "nta"), ("esh", "yta"), ("nah", "nta")):
            recs.append(_record(arm, verdict, guess))
    p = tmp_path / "run.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return p


def test_every_section_renders(run_log):
    groups = by_arm(load(run_log))
    for fn in (headline_table, fanout_note, confusion_table, severity_table,
               confidence_gate_table):
        out = fn(groups)
        assert out and "|" in out, fn.__name__


def test_cli_runs_end_to_end(run_log, tmp_path, capsys):
    """The regression this file exists for: main() must reach every section."""
    import sys

    from jevbench.report import main

    svg = tmp_path / "r.svg"
    argv = sys.argv
    # --md is passed explicitly: its default points into the repo, and a test
    # must not write there.
    sys.argv = ["report", str(run_log), "--svg", str(svg), "--md", str(tmp_path / "out.md")]
    try:
        assert main() == 0
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    for heading in ("Headline", "decomposition", "Where it goes wrong",
                    "severity", "Confidence-gated"):
        assert heading in out
    assert svg.exists()


def test_direct_choice_uses_its_own_confidence():
    v, c = predicted(_record("monolithic", "yta", "yta", conf=0.83))
    assert v == "yta" and c == 0.83


def test_composed_verdict_derives_confidence_from_the_nouls():
    """The decomposed arm has no confidence field, so it must synthesise one."""
    v, c = predicted(_record("decomposed", "yta", "yta"))
    assert v == "yta" and 0.0 < c <= 1.0


def test_baseline_is_the_most_common_verdict(run_log):
    recs = [r for r in load(run_log) if r["arm"] == "monolithic"]
    assert baseline_accuracy(recs) == 0.25   # four verdicts, one each


def test_svg_is_wellformed(run_log):
    import xml.etree.ElementTree as ET

    ET.fromstring(reliability_svg(by_arm(load(run_log))))


def test_empty_run_is_rejected(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(SystemExit):
        load(p)


def test_read_mode_needs_no_api_key(monkeypatch, capsys):
    """`demo --read` must never touch the network or require a key."""
    import sys

    import jevbench.demo as demo

    def boom(*a, **k):
        raise AssertionError("read mode must not call the API")

    monkeypatch.setattr(demo, "ask", boom)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    argv = sys.argv
    sys.argv = ["demo", "--read", "-n", "1"]
    try:
        assert demo.main() == 0
    finally:
        sys.argv = argv
    assert "r/AmItheAsshole" in capsys.readouterr().out


def test_report_writes_results_to_disk(run_log, tmp_path):
    """A run's tables must be kept, not just printed and lost."""
    import sys

    from jevbench.report import main

    md, svg = tmp_path / "results.md", tmp_path / "r.svg"
    argv = sys.argv
    sys.argv = ["report", str(run_log), "--svg", str(svg), "--md", str(md)]
    try:
        assert main() == 0
    finally:
        sys.argv = argv
    body = md.read_text()
    assert "Headline" in body and "Confidence-gated" in body
    assert svg.exists()



def test_context_nouls_cannot_inflate_composed_confidence():
    """Regression: confidence must come only from the nouls that decide.

    op_omits at 0.99 while every deciding noul sits at 0.5 must not report a
    confident answer -- that number feeds ECE, the routing table and the chart.
    """
    rec = _record("decomposed", "yta", "yta")
    rec["probs"] = {k: 0.5 for k in rec["probs"]}
    rec["probs"]["stakes_high"] = 0.99
    rec["probs"]["op_omits"] = 0.01
    _, conf = predicted(rec)
    assert conf == 0.0


def test_failed_requests_are_excluded_from_scoring():
    """An API timeout is not a wrong verdict."""
    ok = _record("monolithic", "yta", "yta")
    bad = _record("monolithic", "nta", "nta")
    bad["error"] = "HTTP 529: overloaded"
    groups = by_arm([ok, bad])
    assert len(groups[("~typesafe/jev-latest", "monolithic")]) == 1


def test_failed_requests_still_appear_in_the_ops_table():
    from jevbench.report import ops_table

    ok = _record("monolithic", "yta", "yta")
    bad = _record("monolithic", "nta", "nta")
    bad["error"] = "HTTP 529"
    body = ops_table([ok, bad])
    assert "| 2 | 1 |" in body        # 2 requests, 1 failed


def test_score_indexing_is_decided_once_for_the_whole_run():
    from jevbench.report import renormalise, score_floor

    run = [{"scores": {"severity": {"raw_score": v}}} for v in (0.0, 1.5, 3.0)]
    floor = score_floor(run)
    assert floor == 0.0
    assert renormalise(3.0, floor, 4) == 1.0     # top level reads as the top
    assert renormalise(0.0, floor, 4) == 0.0


def test_cache_smaller_than_requested_is_not_silently_accepted(tmp_path, monkeypatch):
    """A stale 40-post cache must not quietly satisfy -n 200."""
    import json
    from dataclasses import asdict

    from jevbench import data

    small = [data.Item(id=str(i), title="t", text="x" * 500, verdict="nta",
                       score=10, top_comments=[]) for i in range(3)]
    p = tmp_path / "sample.jsonl"
    p.write_text("".join(json.dumps(asdict(i)) + "\n" for i in small))

    monkeypatch.setattr(data, "sample", lambda n, seed: small * 10)
    assert len(data.load_or_fetch(p, n=3)) == 3      # cache is enough
    assert len(data.load_or_fetch(p, n=20)) == 30    # cache too small -> refetched


def test_several_run_logs_merge_into_one_report(run_log, tmp_path):
    """Phase 2 adds an expensive baseline in a separate run; it must merge."""
    import json

    from jevbench.report import check_same_sample

    other = tmp_path / "later.jsonl"
    recs = [_record("monolithic", "yta", "yta", model="anthropic/claude-sonnet-5")]
    other.write_text("".join(json.dumps(r) + "\n" for r in recs))

    merged = load(run_log, other)
    assert len({r["model"] for r in merged}) == 2
    assert check_same_sample(merged) == ""      # same single post id in both


def test_mismatched_samples_are_flagged(run_log, tmp_path):
    from jevbench.report import check_same_sample

    a = _record("monolithic", "yta", "yta")
    b = _record("monolithic", "yta", "yta", model="anthropic/claude-sonnet-5")
    b["item_id"] = "999"
    note = check_same_sample([a, b])
    assert "not all scored on the same posts" in note


def test_baseline_margin_is_in_percentage_points():
    """Regression: a 53% model against a 35% baseline is +18pp, not +0.2pp."""
    recs = [_record("monolithic", "nta", "nta") for _ in range(7)]
    recs += [_record("monolithic", "yta", "nta") for _ in range(3)]
    body = headline_table(by_arm(recs))
    assert "+0.0 pp" in body        # 70% accurate, 70% baseline -> exactly zero
    assert "+0.7 pp" not in body    # the fraction-vs-points bug
