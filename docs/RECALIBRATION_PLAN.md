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

Not yet run.
