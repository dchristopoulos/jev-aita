# Recalibration check: analysis plan

This plan and the code that carries it out ([`recalibrate.py`](../jevbench/recalibrate.py))
are committed and pushed before the analysis is run. The results section is
added afterwards, in a separate commit, without changing anything above it.

## Question

Is part of Jev's gap to Sonnet 5 a calibration problem? In the Part 1 check,
recalibrating Jev's direct answers improved its weighted Brier on development
posts (0.354 → 0.320). If every model gets the same recalibration on the final
2025 posts, does Sonnet still lead?

## What is already known

This is not a blind analysis. The final logs have been scored and published:
raw weighted Brier is 0.344 for Sonnet and 0.369 for Jev, a paired difference of
−0.025 [−0.046, −0.003]. What this plan fixes in advance is the method and how
the result will be read, before any recalibrated number is computed.

## Data

The committed final logs (`runs/final-*.jsonl`): 770 posts, seven model
configurations, no new API calls. Each configuration's four verdict
probabilities are used as saved, normalised to sum to 1. Qwen's 4 malformed
answers have no probabilities and are left out, as in the main results; paired
comparisons use posts where both configurations have a valid answer.

## Method

For each configuration separately:

- **Features:** its four verdict probabilities.
- **Model:** the multinomial logistic regression from Part 1
  ([`steps.py`](../jevbench/steps.py): standardised features, L2 0.1, 400
  gradient steps), with no changes.
- **Weights:** each post weighted so the four verdicts count at their share of
  the 1,839 eligible 2025 posts, the same target as the headline scores.
- **Cross-fitting:** 10 folds, stratified by verdict, fold seed 0. Each post's
  recalibrated forecast comes from a model fitted on the other nine folds.

**Secondary, Jev only:** fit the same regression on Jev's direct answers for
the 300 development posts (`runs/diagnostic/steps-dev-*.jsonl`, monolithic
arm), then apply it unchanged to Jev's 770 final answers. This is recalibration
learned entirely outside the test posts.

## Outcomes

- **Primary:** paired weighted Brier difference, Sonnet minus Jev, after
  recalibration, with a 95% stratified post-bootstrap interval (2,000 draws,
  seed 0, as in the main report).
- **Secondary:** each configuration's weighted Brier before and after
  recalibration, and its ESH and NAH recall after recalibration; the Jev
  transfer result.

The intervals resample out-of-fold losses without refitting, so like the
Part 1 intervals they leave out the uncertainty from the fitting itself.

## How the result will be read

- If the recalibrated Sonnet-minus-Jev interval **excludes zero**, the gap is
  not explained by calibration.
- If it **includes zero**, the gap may be at least partly calibration. That
  does not show the two are equal.
- Any recalibrated improvement is reported together with ESH and NAH recall,
  since in Part 1 the improvement came from almost never predicting those.
- The headline results do not change either way. This is a secondary analysis.

## Results

Run on 2026-09-23 at 10:12 UTC, after the plan was pushed (GitHub records
that push at 10:12:11 UTC, commit `9adaa0a`), with the code as committed.
Output of `python -m jevbench.recalibrate`:

#### Every configuration, recalibrated the same way

| Configuration | Posts | Raw weighted Brier ↓ | Recalibrated ↓ [95% CI] | Change [95% CI] | ESH / NAH recall, raw → recalibrated |
|---|---:|---:|---:|---:|---|
| Jev · direct | 770 | 0.369 | 0.339 [0.324, 0.356] | -0.030 [-0.044, -0.016] | 3% → 0% / 11% → 0% |
| Jev · two questions | 770 | 0.515 | 0.350 [0.336, 0.364] | -0.166 [-0.177, -0.154] | 32% → 0% / 7% → 0% |
| Sonnet 5 · direct | 770 | 0.344 | 0.332 [0.317, 0.350] | -0.012 [-0.023, -0.002] | 2% → 0% / 4% → 0% |
| GPT-5 nano (low effort) · direct | 770 | 0.480 | 0.386 [0.375, 0.398] | -0.094 [-0.114, -0.076] | 8% → 0% / 22% → 0% |
| GPT-5 nano (minimal effort) · direct | 770 | 0.569 | 0.410 [0.405, 0.415] | -0.159 [-0.174, -0.144] | 3% → 0% / 0% → 0% |
| Qwen 3.6 35B-A3B (local) · direct | 766 | 0.410 | 0.387 [0.375, 0.400] | -0.022 [-0.038, -0.007] | 0% → 0% / 0% → 0% |
| Gemma 4 26B-A4B (local) · direct | 770 | 0.588 | 0.352 [0.340, 0.366] | -0.236 [-0.274, -0.196] | 17% → 0% / 42% → 0% |

#### Primary: Sonnet minus Jev, weighted Brier

| | Difference [95% CI] |
|---|---:|
| Raw (the headline) | -0.025 [-0.048, -0.004] |
| Both recalibrated | -0.007 [-0.021, +0.007] |

#### Secondary: Jev recalibrated on the 300 development posts only

| | Weighted Brier ↓ [95% CI] | ESH / NAH recall |
|---|---:|---|
| Jev raw | 0.369 [0.344, 0.398] | 3% / 11% |
| Jev, fit on development posts | 0.337 [0.321, 0.354] | 0% / 0% |
| Sonnet raw | 0.344 [0.321, 0.370] | 2% / 4% |

Sonnet raw minus Jev fit on development posts: +0.008 [-0.013, +0.028]

**Reading, by the rule set above:** the recalibrated Sonnet-minus-Jev interval
includes zero (−0.007, 95% CI −0.021 to +0.007), so **the gap may be at least
partly calibration**. That does not show the two are equal.

- **Recalibration helped every configuration, Sonnet least.** Sonnet improved
  by 0.012, Jev by 0.030. Most of Jev's 0.025 raw gap to Sonnet closes.
- **It works by giving up the rare verdicts.** After recalibration, every
  configuration predicts ESH or NAH for no post (0% recall). Much of each gain
  comes from matching Reddit's 74% NTA mix rather than from sharper judgments
  of individual posts.
- **Recalibration compresses the field.** Gemma goes from last (0.588) to
  0.352, close to Jev's 0.339; GPT-5 nano minimal ends at 0.410, about the
  base-rate forecast (0.415). Much of the raw spread between models was
  calibration.
- **Secondary transfer check:** fitting Jev's regression on 300 separate 2023
  posts gives 0.337 on the 2025 posts, versus 0.369 for Jev's raw answers.
  After this adjustment Jev chooses ESH or NAH for no post. The exploratory
  comparison with raw Sonnet is +0.008 for Sonnet minus Jev (95% CI −0.013 to
  +0.028). Sonnet has no corresponding development answers to recalibrate, so
  this comparison cannot rank the two adjusted workflows.

**A note on the intervals.** The raw paired interval here is [−0.048,
−0.004], while the main report prints [−0.046, −0.003]. Both use 2,000 draws
and seed 0. This script supplies losses in log order; the main report supplies
them in post-ID order, so the seeded resamples differ. Sorting by post ID
reproduces the main interval. The point estimate is identical.

## Post-hoc audit addendum

Added after an external audit of the results above. Nothing above "Results"
has changed. Two passages in "Results" were reworded after the audit (the
secondary transfer bullet and the note on the intervals); the original wording
is in commit `d2ea058`.

**What the regression does.** It is fitted on all four probabilities, so it
can change which verdict a model picks, not only how confident it is. It
changed 107 of Jev's 770 choices and 36 of Sonnet's. Both models' mean Brier
improved on NTA posts (Jev 0.191 → 0.098, Sonnet 0.135 → 0.093) and got worse
on YTA, ESH and NAH posts; all of the weighted improvement comes from NTA. So
this check does not isolate miscalibration as the cause of the narrowed gap.

**What the interval covers.** The primary interval conditions on the ten
fitted recalibrators and resamples their saved out-of-fold losses. It does not
include variation from refitting them, choosing another fold assignment, or
making new model calls. With fold seeds 0, 1, 2 and 3, the Sonnet-minus-Jev
point difference was −0.0071, −0.0086, −0.0082 and −0.0077. These checks
support the reported point estimate, not equivalence, and not the performance
of a fixed recalibrator on new posts.

**Class weights.** The frozen code calculates class weights from all 770
labels before forming training folds, so held-out labels slightly influence
their training weights. Recalculating the weights within each training fold,
with everything else fixed, changed the Sonnet-minus-Jev point difference from
−0.007061 to −0.007081. Future cross-fitting should calculate weights on
training rows only.

**Timing.** GitHub's activity log shows the plan pushed at 10:12:11 UTC and
the results at 10:14:10 UTC. That verifies the order of the pushes. The time
the analysis was run (10:12:22 UTC) comes from the author's own record.
