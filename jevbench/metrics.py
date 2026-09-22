"""Calibration and cost metrics.

The interesting question about a System One model is not "is it accurate" but
"when it says 0.8, is it right 80% of the time". Reddit gives a verdict, not a
probability, so calibration here is the standard confidence kind: bin answers by
how confident the model was, and check whether it was right that often.

These functions take probabilities and targets in [0,1]; nothing here knows what
the task is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def brier(preds: list[float], truth: list[float]) -> float:
    """Mean squared error between predicted and observed probability.

    Lower is better. Works with fractional truth as well as 0/1: a target of 0.6
    contributes (p - 0.6)^2, so confidently saying 1.0 is penalised.
    """
    if len(preds) != len(truth):
        raise ValueError(f"length mismatch: {len(preds)} preds vs {len(truth)} truth")
    if not preds:
        raise ValueError("no predictions")
    return sum((p - t) ** 2 for p, t in zip(preds, truth)) / len(preds)


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    n: int
    mean_pred: float
    mean_truth: float

    @property
    def gap(self) -> float:
        return self.mean_pred - self.mean_truth


def reliability_bins(preds: list[float], truth: list[float], n_bins: int = 10) -> list[Bin]:
    """Group predictions into equal-width probability bins.

    A perfectly calibrated model has mean_pred == mean_truth in every bin, which
    plots as the diagonal. Empty bins are dropped rather than reported as zero,
    which would fake calibration the model never demonstrated.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for p, t in zip(preds, truth):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"prediction outside [0,1]: {p}")
        # min() keeps p == 1.0 in the last bin rather than overflowing.
        buckets[min(int(p * n_bins), n_bins - 1)].append((p, t))

    out = []
    for i, b in enumerate(buckets):
        if not b:
            continue
        out.append(
            Bin(
                lo=i / n_bins,
                hi=(i + 1) / n_bins,
                n=len(b),
                mean_pred=sum(p for p, _ in b) / len(b),
                mean_truth=sum(t for _, t in b) / len(b),
            )
        )
    return out


def ece(preds: list[float], truth: list[float], n_bins: int = 10) -> float:
    """Expected Calibration Error: sample-weighted mean |mean_pred - mean_truth|.

    0.0 is perfect. Reported alongside the reliability diagram because ECE alone
    hides *where* a model is miscalibrated -- over-confidence at the top end and
    under-confidence at the bottom can cancel into a flattering number.
    """
    bins = reliability_bins(preds, truth, n_bins)
    total = sum(b.n for b in bins)
    if not total:
        raise ValueError("no predictions")
    return sum(b.n * abs(b.gap) for b in bins) / total


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile. q in [0,1]."""
    if not values:
        raise ValueError("no values")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q outside [0,1]: {q}")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


@dataclass(frozen=True)
class Coverage:
    threshold: float
    coverage: float   # fraction of items decided automatically
    accuracy: float   # accuracy on those items only


def cost_usd(prompt_tokens: int, completion_tokens: int, in_per_m: float, out_per_m: float) -> float:
    """Dollar cost of a call. Jev bills output at $0.00, which is the whole point."""
    return (prompt_tokens * in_per_m + completion_tokens * out_per_m) / 1_000_000


def gated_coverage(
    confidence: list[float], correct: list[bool], thresholds: list[float] | None = None
) -> list[Coverage]:
    """Confidence-gated routing using a real confidence signal.

    `coverage_curve` infers certainty from how far a noul sits from 0.5, which
    is the best a Noul allows. Choice and Score return an actual `confidence`
    field, so this takes it directly -- the difference between the two is worth
    reporting, since it is the practical reason to reach for a Choice.
    """
    if len(confidence) != len(correct):
        raise ValueError(f"length mismatch: {len(confidence)} vs {len(correct)}")
    if thresholds is None:
        thresholds = [0.5, 0.7, 0.9, 0.95]
    out = []
    for th in thresholds:
        acted = [c for conf, c in zip(confidence, correct) if conf >= th]
        if not acted:
            out.append(Coverage(th, 0.0, float("nan")))
            continue
        out.append(Coverage(th, len(acted) / len(correct), sum(acted) / len(acted)))
    return out


def bin_support(preds: list[float], truth: list[float], n_bins: int = 10) -> tuple[int, int]:
    """(bins populated, smallest bin count) behind an ECE number.

    ECE is a weighted mean over bins, so a bin holding two samples moves it as
    if it were a measurement. Printing the support next to the number lets a
    reader discount one that rests on almost nothing.
    """
    bins = reliability_bins(preds, truth, n_bins)
    return len(bins), min((b.n for b in bins), default=0)
