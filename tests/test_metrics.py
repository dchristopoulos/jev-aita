"""Scoring rules and calibration helpers."""

import math

import pytest

from jevbench.metrics import (
    bin_support, cost_usd, ece, gated_coverage, multiclass_scores, percentile,
    reliability_bins,
)


def test_multiclass_scores_use_all_four_probabilities_and_exact_class_weights():
    labels = ("nta", "yta", "esh", "nah")
    predictions = [
        {"nta": 0.8, "yta": 0.1, "esh": 0.05, "nah": 0.05},
        {"nta": 0.6, "yta": 0.2, "esh": 0.1, "nah": 0.1},
    ]
    truth = ["nta", "yta"]
    b, ll = multiclass_scores(predictions, truth, labels)
    assert b == pytest.approx((0.055 + 1.02) / 2)
    assert ll == pytest.approx((-math.log(0.8) - math.log(0.2)) / 2)
    with pytest.raises(ValueError):
        multiclass_scores(predictions, truth, labels,
                          {"nta": 9, "yta": 1, "esh": 1, "nah": 1})
    full_truth = ["nta", "yta", "esh", "nah"]
    full_predictions = predictions + [
        {"nta": 0, "yta": 0, "esh": 1, "nah": 0},
        {"nta": 0, "yta": 0, "esh": 0, "nah": 1},
    ]
    wb, _ = multiclass_scores(full_predictions, full_truth, labels,
                              {"nta": 9, "yta": 1, "esh": 1, "nah": 1})
    assert wb == pytest.approx((9 * 0.055 + 1.02) / 12)


def test_multiclass_rejects_incomplete_distribution():
    with pytest.raises(ValueError):
        multiclass_scores([{"nta": 1.0}], ["nta"], ("nta", "yta", "esh", "nah"))


def test_multiclass_log_loss_handles_zero_true_probability():
    _, loss = multiclass_scores([{"nta": 0, "yta": 1, "esh": 0, "nah": 0}],
                                ["nta"], ("nta", "yta", "esh", "nah"))
    assert loss == pytest.approx(-math.log(1e-15))


def test_perfectly_calibrated_model_has_zero_ece():
    preds = [0.05, 0.25, 0.75, 0.95]
    assert ece(preds, preds) == 0.0


def test_ece_catches_uniform_overconfidence():
    # Always claims 0.9, actually right 50% of the time -> gap of 0.4.
    assert ece([0.9] * 10, [0.5] * 10) == pytest.approx(0.4)


def test_ece_hides_cancelling_errors():
    # Documents the known blind spot: over- and under-confidence in *different*
    # bins do not cancel, because ECE takes absolute value per bin.
    preds = [0.9, 0.1]
    truth = [0.5, 0.5]
    assert ece(preds, truth) == pytest.approx(0.4)


def test_probability_of_one_lands_in_last_bin():
    bins = reliability_bins([1.0], [1.0], n_bins=10)
    assert len(bins) == 1 and bins[0].lo == 0.9


def test_empty_bins_are_dropped_not_reported_as_calibrated():
    bins = reliability_bins([0.05, 0.95], [0.0, 1.0], n_bins=10)
    assert len(bins) == 2


def test_reliability_rejects_out_of_range():
    with pytest.raises(ValueError):
        reliability_bins([1.5], [1.0])


def test_percentile_interpolates():
    assert percentile([0.0, 10.0], 0.5) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0


def test_percentile_single_value():
    assert percentile([7.0], 0.95) == 7.0




def test_free_output_tokens_are_free():
    # Jev's pricing shape: output is $0.00/M, so a long answer costs nothing.
    assert cost_usd(1_000_000, 999, in_per_m=0.042, out_per_m=0.0) == pytest.approx(0.042)


def test_gated_coverage_uses_real_confidence():
    conf = [0.99, 0.95, 0.60, 0.40]
    correct = [True, True, False, False]
    curve = {c.threshold: c for c in gated_coverage(conf, correct, [0.5, 0.9])}
    assert curve[0.9].coverage == 0.5 and curve[0.9].accuracy == 1.0
    assert curve[0.5].coverage == 0.75 and curve[0.5].accuracy == pytest.approx(2 / 3)


def test_gated_coverage_reports_nan_when_nothing_clears_the_bar():
    c = gated_coverage([0.1], [True], [0.9])[0]
    assert c.coverage == 0.0 and math.isnan(c.accuracy)


def test_gated_coverage_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        gated_coverage([0.9], [True, False])


def test_bin_support_reports_the_thinnest_bin():
    """An ECE resting on a two-sample bin should be readable as such."""
    n_bins, min_n = bin_support([0.05] * 20 + [0.95, 0.95], [0.0] * 20 + [1.0, 1.0])
    assert n_bins == 2 and min_n == 2
