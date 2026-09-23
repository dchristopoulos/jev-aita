# Five-question follow-up plan

Written before the new API calls. The 2025 labels and all original benchmark
results are already known. The five-question setup was selected after a
300-post 2023 check, so this is a post-hoc follow-up, not a new holdout or a
revision of the original Part 2 result.

## Run

- Use the same frozen 770-post 2025 sample, in the same order, with no new
  filtering. Rebuild it from the pinned source record if needed.
- Ask `~typesafe/jev-latest` the unchanged `yesno5` questions in
  `jevbench/steps.py`, through OpenRouter, one request per post. Use
  `jevbench.bench` and retain its JSONL log and manifest. Record the resolved
  model, failures, retries, latency and billed cost. A model-version change
  from the original run is a limitation, not a reason to discard the run.
- Require one usable answer for every post before scoring the primary result.
  Report an incomplete run as incomplete; do not score only the successes.

Run and score with:

```bash
python -m jevbench.bench --data data/final-ucb-2025.jsonl --models '~typesafe/jev-latest' --arms yesno5 --out runs/followup-yesno5-2025.jsonl --budget 0.10
python -m jevbench.five_question_followup runs/followup-yesno5-2025.jsonl
```

## Analysis fixed before the run

- Fit two separate four-class multinomial logistic regressions on the saved
  300-post 2023 development answers: one on direct Choice probabilities and
  one on the five yes/no probabilities. Use `steps.features` and
  `steps.fit_predict` unchanged: standardise from the training data, L2 = 0.1,
  400 gradient steps, learning rate = 0.5. Weight each training verdict to
  the eligible 2025 source mix. Neither regression sees a 2025 label.
- Primary comparison: five-question fitted probabilities minus direct fitted
  probabilities on the same 770 posts. Score the four-class Brier loss,
  weighted to the eligible 2025 verdict mix. Use the existing 2,000-draw,
  seed-0 paired stratified post bootstrap for a 95% interval. Negative favors
  five questions. If the interval is wholly below zero, call it evidence of
  improvement; wholly above zero, evidence of harm; otherwise, inconclusive.
- Also report weighted top-1 accuracy, macro recall, ESH and NAH recall,
  median request latency, cost per 1,000 posts, and the raw direct row for
  context. The existing Part 2 table contains Sonnet's raw row. Keep raw and
  fitted results visibly distinct. The bootstrap interval conditions on the
  2023 fitted regressions and does not
  measure uncertainty from refitting them or from repeated API calls.

## Results

Run on 2026-09-23 after the plan and the pre-run clarification were pushed.
The [log](../runs/followup-yesno5-2025.jsonl) has 770 successful calls, no
failures and no early stop. The [manifest](../runs/followup-yesno5-2025.meta.json)
records a clean tree at `4958c82`, the original sample hash
`2f8cae7c7bebe3c259efbe69664b463919a9db5b0db386d23a79d8892f82479d`,
and the exact questions. Every record names resolved model
`typesafe/jev-1.13-20260917`. Total billed cost was $0.03362.

| Jev setup | Weighted Brier ↓ [95% CI] | Weighted top-1 [95% CI] | Macro recall | Median call | $ per 1,000 posts |
|---|---:|---:|---:|---:|---:|
| Direct, raw | 0.369 [0.344, 0.398] | 75.4% [72.7%, 77.8%] | 37.4% | 0.39 s | $0.037 |
| Direct, fitted on 2023 | 0.337 [0.321, 0.354] | 78.6% [77.0%, 80.1%] | 33.0% | 0.39 s | $0.037 |
| Five yes/no, fitted on 2023 | 0.346 [0.332, 0.361] | 76.4% [74.8%, 77.8%] | 30.1% | 0.57 s | $0.044 |

Primary paired weighted Brier difference, five questions minus fitted direct:
**+0.009 [−0.001, +0.019]**. The interval crosses zero, so the planned rule
calls this inconclusive. ESH and NAH recall were 0% for both fitted setups.
Five questions did not establish an advantage over direct, and the original
raw-answer benchmark remains unchanged.

The bootstrap conditions on the two regressions fitted to 2023 posts. It does
not include uncertainty from refitting them or from repeated API calls. The
2025 labels and original results were already known when this setup was
chosen. Latency for five questions was measured on a later run, so its median
is not a controlled speed comparison with direct.

## Pre-run clarification

Use `python3` in the commands above on systems without a `python` alias. This
changes the executable name, not the test or analysis. The scorer now checks
the saved manifest against the original sample hash and five-question wording
and prints rare-verdict recall for both fitted setups. These checks do not
change the planned outcome or decision rule.
