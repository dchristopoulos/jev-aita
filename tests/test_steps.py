"""The step setups and their out-of-fold combiner, without network."""

import random

from jevbench.bench import ARMS, JEV_ONLY
from jevbench.client import _split_answers
from jevbench.steps import ARMS as STEP_ARMS, LABELS, features, out_of_fold


def test_every_step_question_is_well_formed_and_jev_only():
    for arm, make in STEP_ARMS.items():
        assert arm in ARMS and arm in JEV_ONLY
        for q in make().values():
            assert q["type"] in {"noul", "choice", "score"} and q["instructions"].endswith("?")


def test_score_answers_are_read_as_level_probabilities():
    q = {"sev": {"type": "score", "instructions": "How bad?", "criteria": ["a", "b", "c"]}}
    a = {"sev": {"score": 1.1, "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2}, "confidence": 0.7}}
    _, choices = _split_answers(a, q)
    assert choices["sev"].probabilities == {"0": 0.1, "1": 0.7, "2": 0.2}


def test_features_cover_yes_no_and_option_answers_in_a_fixed_order():
    rec = {"probs": {"b": 0.2, "a": 0.9}, "choices": {"s": {"probabilities": {"1": 0.3, "0": 0.7}}}}
    assert features(rec) == [0.9, 0.2, 0.7, 0.3]


def test_out_of_fold_fit_learns_a_signal_it_was_not_shown():
    rng = random.Random(1)
    y = [i % 4 for i in range(80)]
    x = [[float(c == k) + rng.gauss(0, 0.3) for k in range(4)] for c in y]
    preds = out_of_fold(x, y, [1.0] * len(y))
    hits = sum(max(p, key=p.get) == LABELS[c] for p, c in zip(preds, y))
    assert hits >= 70
