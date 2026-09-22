# Final AITA run protocol

This is the analysis plan for the 770-post holdout. I did not commit it to git
before the final runs started, so its timing cannot be proven from the
history. The plan sections below were last changed at 23:00 on 22 September
2026, while the chat runs were still in progress and after the Jev run had
finished; this note and the deviations were added afterwards. Every change
from the plan made after the runs began is listed under
[Deviations](#deviations).

## Data and target

Use the `dilemmas` split of `ucberkeley-dlab/fragility-moral-judgment-llms` at revision
`cb4c298cbfa93ce9cdf56685f12a55b0a6928110`. The posts date from January to March
2025. The official post flair is the target. It reflects one top-voted comment at the
flair cutoff, so it is not a vote majority or a moral fact.

The frozen sample has 365 NTA, 285 YTA, 65 ESH, and 55 NAH posts. Selection used a
fixed seed, 400 to 4,000 characters of cleaned post text, no karma cutoff, and a
filter for verdict phrases in the title or post. All posts have 25 to 99 parsed
top-level comment verdicts in the source dataset. The 770 IDs do not overlap the
old 200-post development set or the 40-post pipeline pilot.

## Models and prompts

Jev receives one native Choice question with the four label definitions. Its
two-question arm asks whether the poster and another person merit blame; the
four probabilities come from the product of those two answers. This product
assumes independence and is a baseline for decomposition, not a calibrated
claim. Chat models receive a normal classifier system prompt with the same
label meanings and return all four probabilities. Each model sees only the
title and post. Nano runs with minimal and low reasoning. Qwen and Gemma run
locally with reasoning off. The chat parser accepts four finite nonnegative
numbers and normalizes their sum. A missing label, invalid number, or all-zero
vector remains a malformed answer.

## Analysis

The primary measure is multiclass Brier score, weighted by the four class
frequencies among all eligible source posts: 1,361 NTA, 342 YTA, 71 ESH, and
65 NAH. Compare models on paired post IDs. Report a paired bootstrap interval
for each difference from Jev. Also report unweighted Brier, log loss, top-1
accuracy, recall by class, complete-distribution rate, latency, and billed API
cost. Include two no-model baselines: always NTA for accuracy and the eligible
source class frequencies as a constant probability distribution for Brier and
log loss. The fixed 40-post pilot is a pipeline check and contributes no result
to the final estimates.

As a label-quality check, repeat accuracy and Brier on posts whose official
flair agrees with the plurality of vote-weighted parsed comment verdicts among
the four target labels; INFO is excluded from that comparison.
Report the subset size by class. This check uses the source comments only for
analysis; the models never receive them. Show the size of the mismatch between
flair and both vote-weighted and unweighted comment pluralities.

These public posts may have appeared in model training. Their 2025 dates
reduce the risk compared with the old 2023 source, but cannot rule it out.

## Deviations

All times are 22 September 2026, local time (UTC+3), taken from the request
timestamps in the logs.

1. **The chat parser was loosened, and two runs were restarted.** All runs
   started at 22:38–22:39. The first local Qwen run had 6 malformed answers in
   184 posts and the first nano low-effort run had 1 in 104. Raw replies are
   not logged; the code comment written with the fix describes them as
   relative weights that did not sum to 1, such as 0.1, 0.85, 0.1, 0.05. I
   stopped both runs, changed the parser to rescale any four finite
   nonnegative numbers with a positive sum, and restarted both from the first
   post: local models at 22:45, nano low at 22:48. The parser rule stated
   above is the new one. Sonnet and nano minimal were already running under
   the old parser and had no malformed answers, and Jev does not use the chat
   parser, so the change could only affect the two restarted runs. The prompt
   did not change: per-post prompt token counts are identical between the
   stopped and restarted runs. The stopped logs are kept in
   `runs/diagnostic/` and are not used.
2. **A label-only check was started and stopped.** At 22:53 I began an
   exploratory chat prompt that asks for one numbered verdict instead of four
   probabilities, with the same label definitions. I ran it on GPT-5 nano with
   minimal effort only: 55 posts, then 83 posts in a second attempt saved as
   `standard-choice-api-tolerant-2025`. I stopped when I lowered the spending
   limit. It is not part of the comparison. Requiring a
   probability vector therefore remains a limitation when judging top-1
   accuracy, latency and cost.
3. **The final runs came from an uncommitted working tree, which is not
   published.** Their manifests record `git_sha` `e473a91`, which does not
   identify the code. The `code_sha256` field does, but neither of its two
   values matches the published code, which has since been cleaned up. What
   can be checked:
   - The chat prompt: `scripts/check_prompt_tokens.py` shows the published
     prompt matches the billed prompt token count of every GPT-5 nano call.
   - Jev's direct question and the four label definitions are identical to
     commit `e473a91`, made before the runs (`git show e473a91:jevbench/questions.py`).
   - Jev's two-question text was added after that commit. Its file was last
     edited at 22:37, before the first final request, but that timestamp is
     not recorded anywhere in the repo.
4. **Reporting additions after the runs.** After the results were in, I added
   confidence intervals for weighted Brier and weighted top-1, macro recall,
   and paired intervals for weighted log loss and unweighted Brier. These are
   extra views of the same predictions, added in response to review; they do
   not change the primary measure or any prediction. I also stopped plotting
   malformed answers in the calibration chart and its ECE, where they had
   appeared as a bin at probability 0. This moves Qwen's ECE from 0.287 to
   0.289, changes its ECE-support cell, and changes no other number.
   A later review led me to show always-NTA's log loss with the same 1e-15
   clip every model gets (8.977, previously shown as ∞), and to label the
   calibration results as unweighted sample figures. No other number changed.
