from collections import Counter

import pytest

from jevbench.data import VERDICTS
from jevbench.questions import (
    CONTEXT, DIMENSIONS, OP_FAULT, OTHER_FAULT, compose_verdict, decomposed,
    full, monolithic, severity_target,
)


def test_arms_grow_without_losing_the_verdict_question():
    assert len(monolithic()) == 1
    assert len(decomposed()) == len(DIMENSIONS) == 8
    assert len(full()) == 10


def test_full_arm_uses_all_three_primitives():
    assert Counter(q["type"] for q in full().values()) == {"noul": 8, "choice": 1, "score": 1}


def test_decomposed_arm_never_mentions_the_verdict():
    """The point of the arm: the model makes factual judgments, code judges."""
    for qid, q in decomposed().items():
        assert q["type"] == "noul", qid
        blob = (q["instructions"] + " ".join(q["criteria"].values())).lower()
        for word in ("asshole", "verdict", "yta", "nta"):
            assert word not in blob, f"{qid} leaks the verdict via {word!r}"


def test_choice_offers_exactly_the_four_verdicts():
    assert set(monolithic()["verdict"]["criteria"]) == set(VERDICTS) == {
        "yta", "nta", "esh", "nah"}


def test_every_noul_is_well_formed():
    for qid, q in decomposed().items():
        assert q["instructions"].endswith("?"), qid
        assert set(q["criteria"]) == {"true", "false"}, qid


@pytest.mark.parametrize(
    "probs,expected",
    [
        ({"op_broke_agreement": 0.9, "other_behaved_badly": 0.1}, "yta"),
        ({"op_disproportionate": 0.8, "other_escalated": 0.7}, "esh"),
        ({"op_deceived": 0.1, "other_behaved_badly": 0.9}, "nta"),
        ({"op_deceived": 0.1, "other_behaved_badly": 0.2}, "nah"),
    ],
)
def test_composition_covers_all_four_verdicts(probs, expected):
    assert compose_verdict(probs) == expected


def test_composition_is_disjunctive_not_averaging():
    """One damning answer must not be buried by three mild ones."""
    probs = {"op_broke_agreement": 0.95, "op_disproportionate": 0.0,
             "op_disregarded": 0.0, "op_deceived": 0.0, "other_behaved_badly": 0.0}
    assert compose_verdict(probs) == "yta"


def test_context_questions_do_not_decide_the_verdict():
    """stakes_high and op_omits inform a reader; they must not swing the fold."""
    quiet = {"op_deceived": 0.1, "other_behaved_badly": 0.1}
    loud = {**quiet, **{k: 0.99 for k in CONTEXT}}
    assert compose_verdict(quiet) == compose_verdict(loud) == "nah"


def test_sides_are_disjoint():
    assert not set(OP_FAULT) & set(OTHER_FAULT)


def test_composition_handles_missing_answers():
    assert compose_verdict({}) == "nah"


def test_severity_target_orders_the_verdicts():
    assert severity_target("nah") == severity_target("nta") == 0.0
    assert severity_target("yta") == severity_target("esh") > 0.0
