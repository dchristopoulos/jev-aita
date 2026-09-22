import pytest

from jevbench.data import VERDICTS
from jevbench.questions import balanced, balanced_distribution, monolithic


def test_direct_arm_is_one_choice_over_exactly_the_four_verdicts():
    q = monolithic()
    assert list(q) == ["verdict"] and q["verdict"]["type"] == "choice"
    assert q["verdict"]["criteria"] == VERDICTS


def test_two_question_arm_is_two_well_formed_nouls():
    q = balanced()
    assert set(q) == {"poster_at_fault", "other_at_fault"}
    for spec in q.values():
        assert spec["type"] == "noul" and spec["instructions"].endswith("?")
        assert set(spec["criteria"]) == {"true", "false"}


def test_two_answers_make_a_four_verdict_distribution():
    p = balanced_distribution({"poster_at_fault": 0.8, "other_at_fault": 0.2})
    assert p == pytest.approx({"yta": 0.64, "nta": 0.04, "esh": 0.16, "nah": 0.16})
    assert sum(p.values()) == pytest.approx(1)


def test_out_of_range_fault_probability_is_rejected():
    with pytest.raises(ValueError):
        balanced_distribution({"poster_at_fault": 1.2, "other_at_fault": 0.2})
