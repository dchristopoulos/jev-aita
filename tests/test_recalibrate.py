"""The recalibration check on synthetic forecasts, without the real logs."""

import random

from jevbench.recalibrate import cross_fitted, paired, score, transferred

PRIORS = {"nta": 70, "yta": 20, "esh": 5, "nah": 5}


def _rows(n, seed):
    """An informative but badly calibrated forecaster: always 40% on the truth,
    even when the truth is NTA, which is common enough to deserve more."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        t = rng.choice(["nta"] * 5 + ["yta"] * 3 + ["esh", "nah"])
        rows.append((f"p{seed}-{i}", t, {v: 0.4 if v == t else 0.2 for v in PRIORS}))
    return rows


def test_recalibration_fixes_a_miscalibrated_forecaster_out_of_fold():
    rows = _rows(200, 0)
    fixed = cross_fitted(rows, PRIORS)
    assert [i for i, _, _ in fixed] == [i for i, _, _ in rows]
    assert all(abs(sum(p.values()) - 1) < 1e-9 for _, _, p in fixed)
    assert score(fixed, PRIORS)[0] < score(rows, PRIORS)[0] - 0.1
    assert paired(fixed, rows, PRIORS)[2] < 0


def test_a_fit_on_one_sample_transfers_to_another():
    assert score(transferred(_rows(200, 1), _rows(200, 2), PRIORS), PRIORS)[0] < \
        score(_rows(200, 2), PRIORS)[0] - 0.1
