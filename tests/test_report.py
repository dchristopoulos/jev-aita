"""Report tests on hand-built run logs, so no API key or network is needed.

The end-to-end test catches the class of bug where a section function exists
but is unreachable from main() -- every table passes its own unit test and the
CLI still crashes.
"""

import json
import sys
import xml.etree.ElementTree as ET

import pytest

from jevbench.client import UNPARSED_CHOICE
from jevbench.report import (
    baseline_accuracy, by_arm, check_same_sample, confusion_table, consensus_check,
    constant_baselines, distribution, headline_svg, headline_table, load, macro_recall, main,
    ops_table, paired_table, predicted, probability_table, reliability_svg, stratified,
)

JEV, CHAT = "~typesafe/jev-latest", "anthropic/claude-sonnet-5"
PRIORS = {"nta": 7, "yta": 1, "esh": 1, "nah": 1}


def _record(verdict, guess, item="1", model=JEV, arm="monolithic", p=0.7):
    rec = {"model": model, "arm": arm, "item_id": item, "n_questions": 1,
           "probs": {}, "choices": {}, "verdict_true": verdict, "post_score": 0,
           "latency_s": 0.2, "prompt_tokens": 500, "completion_tokens": 0,
           "total_tokens": 500, "cost_usd": 2.1e-5, "parse_failed": False}
    if arm == "balanced":
        rec["n_questions"] = 2
        rec["probs"] = {"poster_at_fault": 0.8 if guess in ("yta", "esh") else 0.2,
                        "other_at_fault": 0.8 if guess in ("nta", "esh") else 0.2}
    else:
        rest = (1 - p) / 3
        rec["choices"] = {"verdict": {"choice": guess, "confidence": p, "probabilities": {
            v: p if v == guess else rest for v in ("yta", "nta", "esh", "nah")}}}
    return rec


def _run(model=JEV, arm="monolithic", right=True):
    """Two posts per class; the model gets one of each right, or all of them."""
    recs = []
    for i, v in enumerate(["yta", "yta", "nta", "nta", "esh", "esh", "nah", "nah"]):
        guess = v if right or i % 2 == 0 else "nta"
        recs.append(_record(v, guess, item=str(i), model=model, arm=arm))
    return recs


@pytest.fixture
def run_log(tmp_path):
    recs = _run() + _run(arm="balanced") + _run(model=CHAT, right=False)
    p = tmp_path / "run.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return p


def test_cli_writes_every_section_and_both_charts(run_log, tmp_path, capsys):
    priors = tmp_path / "source.json"
    priors.write_text(json.dumps({"eligible_mix": PRIORS}))
    md, svg, chart = tmp_path / "out.md", tmp_path / "r.svg", tmp_path / "h.svg"
    argv = sys.argv
    # --md is passed explicitly: its default points into the repo, and a test
    # must not write there.
    sys.argv = ["report", str(run_log), "--svg", str(svg), "--md", str(md),
                "--chart", str(chart), "--priors", str(priors)]
    try:
        assert main() == 0
    finally:
        sys.argv = argv
    body = md.read_text()
    for heading in ("Primary score", "Paired comparison", "No-model baselines",
                    "Accuracy, latency and cost", "Two-question", "Where it goes wrong",
                    "Confidence-gated", "Cost, speed and failures"):
        assert heading in body, heading
    assert "Jev · direct" in body and "Sonnet 5 · direct" in body
    ET.fromstring(svg.read_text())
    ET.fromstring(chart.read_text())
    assert "no-model forecast" in chart.read_text()


def test_chart_requires_priors(run_log, tmp_path):
    argv = sys.argv
    sys.argv = ["report", str(run_log), "--svg", str(tmp_path / "r.svg"),
                "--md", str(tmp_path / "o.md"), "--chart", str(tmp_path / "h.svg")]
    try:
        with pytest.raises(SystemExit):
            main()
    finally:
        sys.argv = argv


def test_empty_run_is_rejected(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(SystemExit):
        load(p)


def test_direct_choice_uses_chosen_label_probability():
    assert predicted(_record("nta", "nta", p=0.7)) == ("nta", pytest.approx(0.7))


def test_direct_choice_falls_back_to_native_confidence_without_distribution():
    rec = _record("yta", "yta", p=0.83)
    rec["choices"]["verdict"]["probabilities"] = {}
    assert predicted(rec) == ("yta", 0.83) and distribution(rec) is None


def test_two_question_arm_has_a_scored_distribution():
    rec = _record("yta", "yta", arm="balanced")
    assert distribution(rec)["yta"] == pytest.approx(0.64)
    assert predicted(rec)[0] == "yta"


def test_rounded_choice_distribution_is_normalised():
    rec = _record("nta", "nta")
    rec["choices"]["verdict"]["probabilities"] = {"nta": 0.56, "yta": 0.02, "esh": 0.01, "nah": 0.40}
    assert sum(distribution(rec).values()) == pytest.approx(1.0)


def test_baseline_margin_is_in_percentage_points():
    """Regression: a 53% model against a 35% baseline is +18pp, not +0.2pp."""
    recs = [_record("nta", "nta", item=str(i)) for i in range(7)]
    recs += [_record("yta", "nta", item=str(i + 7)) for i in range(3)]
    assert baseline_accuracy(recs) == 0.7
    body = headline_table(by_arm(recs))
    assert "+0.0 pp" in body and "+0.7 pp" not in body


def test_macro_recall_gives_always_nta_a_quarter():
    recs = [_record(v, "nta", item=v) for v in ("yta", "nta", "esh", "nah")]
    assert macro_recall(recs) == 0.25


def test_stratified_mean_is_prior_weighted_and_its_interval_brackets_it():
    parts = {"nta": [0.0, 0.0], "yta": [1.0, 1.0], "esh": [1.0, 0.0], "nah": [0.5, 0.5]}
    mean, lo, hi = stratified(parts, PRIORS)
    assert mean == pytest.approx((7 * 0 + 1 * 1 + 1 * 0.5 + 1 * 0.5) / 10)
    assert lo <= mean <= hi
    assert stratified(parts, PRIORS) == (mean, lo, hi)          # seeded, reproducible
    assert stratified(parts, None)[0] == pytest.approx(4 / 8)   # pooled without priors


def test_probability_table_shows_coverage_and_weighted_intervals():
    recs = _run()
    recs[0]["choices"]["verdict"]["probabilities"] = {}
    table = probability_table(by_arm(recs), PRIORS)
    assert "| Jev · direct | 7/8 |" in table and "[" in table


def test_paired_table_shows_three_views_and_refuses_different_samples():
    groups = by_arm(_run() + _run(model=CHAT, right=False))
    table = paired_table(groups, PRIORS)
    assert "Weighted log loss Δ" in table and "Unweighted Brier Δ" in table
    row = next(line for line in table.splitlines() if line.startswith("| Sonnet"))
    assert row.count("[+") == 3         # the half-wrong model is worse on every view
    other = _run(model=CHAT)
    other[0]["item_id"] = "different"
    assert "samples or distributions differ" in paired_table(by_arm(_run() + other), PRIORS)


def test_constant_population_baselines_use_source_priors():
    table = constant_baselines(_run(), PRIORS)
    assert "| Always NTA | 0.600 |" in table and "70.0%" in table


def test_failed_requests_are_excluded_from_scoring_but_counted_in_ops():
    ok, bad = _record("yta", "yta"), _record("nta", "nta", item="2")
    bad["error"] = "HTTP 529: overloaded"
    assert len(by_arm([ok, bad])[(JEV, "monolithic")]) == 1
    assert "| 2 | 1 |" in ops_table([ok, bad])        # 2 requests, 1 failed


def test_several_run_logs_merge_and_mismatched_samples_are_flagged(run_log, tmp_path):
    other = tmp_path / "later.jsonl"
    other.write_text(json.dumps(_record("yta", "yta", item="0", model="x")) + "\n")
    assert len({r["model"] for r in load(run_log, other)}) == 3
    assert "not all scored on the same posts" in check_same_sample(load(run_log, other))
    assert check_same_sample(load(run_log)) == ""


def test_consensus_check_excludes_info_and_flags_unweighted_ties():
    rec = _record("nta", "nta")
    source = [{"id": rec["item_id"],
               **{f"comments_prop_{v.upper()}": p for v, p in
                  {"nta": .4, "yta": .4, "esh": .1, "nah": .1}.items()},
               **{f"comments_prop_weighted_{v.upper()}": p for v, p in
                  {"nta": .6, "yta": .2, "esh": .1, "nah": .1}.items()}}]
    result = consensus_check(by_arm([rec]), source)
    assert "vote-weighted four-verdict plurality on 1/1" in result
    assert "unweighted plurality matches on 0/1 posts, with 1 ties" in result


def test_confusion_table_keeps_unparseable_answers_in_row_totals():
    rec = _record("nta", "nta")
    rec["choices"]["verdict"] = {"choice": UNPARSED_CHOICE, "confidence": 0.0, "probabilities": {}}
    table = confusion_table(by_arm([rec]))
    assert "| **NTA** | 0 | 0 | 0 | 0 | 1 | 0% |" in table


def test_charts_are_wellformed_svg():
    groups = by_arm(_run() + _run(model=CHAT, right=False))
    ET.fromstring(reliability_svg(groups))
    ET.fromstring(headline_svg(groups, PRIORS))


def test_malformed_answers_are_wrong_but_not_calibrated():
    recs = [_record("nta", "nta", item=str(i), p=1.0) for i in range(3)]
    recs[0]["choices"]["verdict"] = {"choice": UNPARSED_CHOICE, "confidence": 0.0,
                                     "probabilities": {}}
    row = headline_table(by_arm(recs)).splitlines()[-1]
    assert "| 66.7% |" in row and "| 0.000 |" in row   # 2 of 3 right, perfect ECE on the rest


def test_new_malformed_answers_score_as_uniform_but_old_ones_stay_excluded():
    new = {"arm": "monolithic", "parse_failed": True, "raw_output": "garbage",
           "choices": {"verdict": {"choice": UNPARSED_CHOICE, "probabilities": {}}}}
    old = {k: v for k, v in new.items() if k != "raw_output"}
    assert distribution(new) == {"yta": 0.25, "nta": 0.25, "esh": 0.25, "nah": 0.25}
    assert distribution(old) is None


def test_report_note_describes_the_malformed_policy_the_logs_actually_get():
    from pathlib import Path
    from jevbench.report import render
    old = [_record(v, v, item=str(i)) for i, v in enumerate(["nta", "yta", "esh", "nah"])]
    new = [dict(r, raw_output="x") for r in old]
    note = lambda recs: render(by_arm(recs), recs, Path("r.svg"), priors=PRIORS)
    assert "left out of Brier and log loss" in note(old)
    assert "uniform forecast" in note(new) and "uniform forecast" not in note(old)
