"""The follow-up analysis works on the saved logs without making API calls."""

import json

from jevbench.five_question_followup import DEV, DIRECT, PRIORS, arm_rows, evaluate
from jevbench.report import load


def test_analysis_reproduces_the_raw_direct_row_with_synthetic_five_question_answers():
    dev = load(*DEV)
    direct = load(DIRECT)
    examples = [r for r in dev if r["arm"] == "yesno5"]
    originals = [r for r in direct if r["arm"] == "monolithic"]
    # Synthetic values exercise the analysis; they are not a five-question result.
    five = [{**examples[n % len(examples)], "item_id": r["item_id"],
             "verdict_true": r["verdict_true"]} for n, r in enumerate(originals)]
    table = evaluate(dev, direct, five, json.loads(PRIORS.read_text())["eligible_mix"])
    assert "| Direct, raw | 0.369 [0.344, 0.398] | 75.4% [72.7%, 77.8%]" in table
    assert "| Direct, fitted on 2023 | 0.337" in table
    assert "Paired weighted Brier difference" in table


def test_incomplete_answers_are_rejected():
    try:
        arm_rows([{"arm": "yesno5", "item_id": "x", "error": None,
                   "parse_failed": False, "probs": {}}], "yesno5", 1)
    except ValueError as e:
        assert "missing or invalid" in str(e)
    else:
        raise AssertionError("incomplete answer was accepted")
