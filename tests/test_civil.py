"""The calibration arm's own checks.

The point of this module is that the target is a *fraction*, not a 0/1 label,
so the tests guard that property rather than re-testing the metrics.
"""
import json
from pathlib import Path

from jevbench.civil import BANDS, TOXIC, Comment, load_or_fetch, report
from jevbench.metrics import brier, ece


def test_bands_cover_the_unit_interval_without_gaps():
    assert BANDS[0][0] == 0.0
    for (_, hi), (lo, _) in zip(BANDS, BANDS[1:]):
        assert hi == lo, "a toxicity value between bands would be silently dropped"
    assert BANDS[-1][1] > 1.0, "toxicity == 1.0 must land in the last band"


def test_the_question_never_leaks_the_answer():
    blob = json.dumps(TOXIC).lower()
    for leak in ("fraction", "rater", "0.6", "percent", "majority"):
        assert leak not in blob


def test_fractional_truth_is_scored_as_a_probability():
    # A model that says 0.6 against a 0.6 rater split is PERFECT here, though a
    # binary-label benchmark would score it wrong 40% of the time.
    assert brier([0.6], [0.6]) == 0.0
    assert ece([0.6], [0.6]) == 0.0
    assert brier([1.0], [0.6]) > brier([0.6], [0.6])


def test_report_renders_from_a_log(tmp_path: Path):
    log = tmp_path / "civil.jsonl"
    with log.open("w") as fh:
        for i, (p, t) in enumerate([(0.1, 0.0), (0.5, 0.6), (0.9, 0.8)]):
            fh.write(json.dumps({"model": "m", "predicted": p, "toxicity_true": t,
                                 "latency_s": 0.4, "error": None}) + "\n")
    out = report([log])
    assert "| `m` | 3 |" in out and "reliability" in out


def test_error_records_are_excluded(tmp_path: Path):
    log = tmp_path / "civil.jsonl"
    with log.open("w") as fh:
        fh.write(json.dumps({"model": "m", "predicted": 0.5, "toxicity_true": 0.5,
                             "latency_s": 0.4, "error": None}) + "\n")
        fh.write(json.dumps({"model": "m", "predicted": None, "toxicity_true": 0.9,
                             "error": "boom"}) + "\n")
    assert "| `m` | 1 |" in report([log])


def test_cache_is_reused_without_network(monkeypatch, tmp_path):
    import jevbench.civil as civil
    cache = tmp_path / "civil.jsonl"
    per_band = 1
    with cache.open("w") as fh:
        for i in range(per_band * len(BANDS)):
            fh.write(json.dumps({"text": f"c{i}", "toxicity": 0.1 * i}) + "\n")
    monkeypatch.setattr(civil, "CACHE", cache)
    monkeypatch.setattr(civil, "fetch", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("hit the network despite a warm cache")))
    assert len(civil.load_or_fetch(len(BANDS))) == len(BANDS)


def test_scaling_recovers_a_known_distortion():
    # Build predictions that are correctly ORDERED but on the wrong scale, the
    # exact failure the real data shows, and check the fit undoes it.
    import math

    from jevbench.civil import _logit, apply_scaling, fit_scaling

    truth = [i / 50 for i in range(1, 50)]
    skewed = [1 / (1 + math.exp(-(1.8 * _logit(t) + 1.2))) for t in truth]
    assert brier(skewed, truth) > 0.02
    a, b = fit_scaling(skewed, truth)
    assert brier(apply_scaling(skewed, a, b), truth) < 0.005


def test_recalibration_is_scored_out_of_sample(monkeypatch):
    from jevbench.civil import split_half_recalibration
    recs = [{"model": "m", "predicted": 0.9, "toxicity_true": 0.3} for _ in range(20)]
    out = split_half_recalibration(recs)
    assert out[0]["n_fit"] == 10 and out[0]["n_test"] == 10
    assert out[0]["n_fit"] + out[0]["n_test"] == len(recs)
