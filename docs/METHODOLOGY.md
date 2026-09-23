# Methodology and limitations

This benchmark asks a narrow question: how well can Jev and chat models predict
the official verdict flair on a post from Reddit's AITA forum (r/AmItheAsshole)
from its title and text? The flair is a subreddit decision, not a fact about
who behaved well. The [run
protocol](FINAL_PROTOCOL.md) lists the plan and every deviation from it.
[Results](results.md) are generated from the saved request logs.

## Posts and labels

The source is UC Berkeley D-Lab's [2025 dilemma dataset](https://huggingface.co/datasets/ucberkeley-dlab/fragility-moral-judgment-llms),
revision `cb4c298cbfa93ce9cdf56685f12a55b0a6928110`. Its authors
collected 2,939 AITA posts from January through March 2025 and parsed comment
verdicts. I kept posts with one of the four target flairs, 400 to 4,000
characters of cleaned body text, a title, and no verdict phrase in either the
title or body. I did not use post karma as a quality filter. The eligible
population has 1,361 NTA, 342 YTA, 71 ESH, and 65 NAH posts. A fixed seed
selected 365, 285, 65, and 55 respectively. The 770 posts share no IDs with
the earlier development sample or pipeline pilot. Each has 25 to 99 parsed
top-level comment verdicts in the source.

This is a stratified sample so that ESH and NAH can be measured at all. Raw
sample accuracy is not real-world accuracy. For the population estimate I
weight each class by its frequency among all eligible posts. That population
is the filtered 2025 source, not all of Reddit.

The flair is noisy. Among the 770 posts, it agrees with the plurality of the
four vote-weighted comment verdict shares on 704 posts. It agrees with the
unweighted plurality on 613. The unweighted comparison has 13 ties; the
weighted one has none. INFO is excluded because it is not a target class.
These comment aggregates are a label-quality check, not information shown to
the models. A top-rated comment can decide the flair without representing a
majority of other comments. The source [paper](https://arxiv.org/abs/2603.05651)
also finds that models' AITA judgments change substantially with prompt
protocol and point of view. Neither a flair match nor a comment plurality
establishes moral correctness.

The leak filter removes posts that state a verdict, such as "EDIT: I'm NTA".
It does not remove edits written after reading the comments, such as "EDIT: a
lot of people are saying…". A small share of posts contain those. Every model
sees the same text, so they affect all models alike.

## Questions and chat prompts

The primary comparison is between probability-producing configurations, not
between each model's best possible classifier prompt. Top-1 accuracy, latency
and cost from the probability prompt include the cost of generating four
numbers. Jev and the chat models also get slightly different target wording
(below), and the local models are 4-bit quantized builds with reasoning off,
so their results do not stand for full-precision or hosted versions.

Jev's direct arm sends one native Choice question: which verdict would the
subreddit reach? Each of YTA, NTA, ESH, and NAH has a one-line definition.
Its second arm asks two symmetric yes/no questions about blame, then computes
four verdict probabilities from their product. That product assumes the two
answers are independent and is not fitted to data. So the two-question result
says that *this combination rule* is worse, not that asking two questions
carries less information.

The chat models do not receive Jev's yes/no questions. They receive a normal
system prompt asking them to predict the official subreddit flair from the
post, with the same four label definitions, and must return a four-number
probability vector. That allows a proper probability score against Jev's
native distribution. It also costs chat models output tokens, averaging 46 per
Sonnet call, and may lower their top-1 accuracy compared with an ordinary
single-label prompt. [TypeSafe's own write-up](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
says its structured-decision wrapper for LLMs "tends to be slower and more
expensive than giving decisions without probabilities". So the run tests a
structured probabilistic workflow, not the best possible prompt for a one-word
verdict. Jev's question says "which
verdict would the subreddit reach" while the chat prompt says "predict the
official flair". They aim at the same label, but the wording is not identical.

The chat request sends `temperature: 0` and a reasoning effort. GPT-5 nano ran
at "minimal" and "low". Sonnet 5 was sent the runner's default, "low", but
the provider reported zero reasoning tokens on all 770 calls, so it answered
without extended thinking. Qwen and Gemma ran with reasoning off.

The exact prompts are in [`client.py`](../jevbench/client.py) and
[`questions.py`](../jevbench/questions.py). A test pins their SHA-256, and
[`check_prompt_tokens.py`](../scripts/check_prompt_tokens.py) shows that the
billed prompt token count of every GPT-5 nano call equals the tokenized
current chat prompt plus the post, plus a constant 10 tokens of chat framing.
That check covers the chat prompt only; the [protocol](FINAL_PROTOCOL.md#deviations)
lists what is known about the Jev questions. The logs record model
identifiers, tokens, provider cost, latency, retries, errors, and parsed
answers. They do not contain post text or raw replies.

When a chat model gives two labels the same top probability, the first in the
order YTA, NTA, ESH, NAH is taken as its choice. Gemma has 72 such posts and
the other chat models 8 to 15. Splitting credit evenly across tied labels
instead moves weighted top-1 and macro recall by at most 0.5 points for any
arm.

## Scoring

The primary score is four-class Brier loss, weighted to the eligible source
class mix. Lower is better; zero is perfect. A constant forecast using those
source class frequencies is the probability baseline. Always predicting NTA
is the accuracy baseline. I also show log loss, unweighted sample scores,
top-1 accuracy, macro recall (the unweighted mean of the four class recalls,
where always-NTA scores 25%), recall by class, and the fraction of calls that
produced all four usable probabilities.

Intervals come from 2,000 bootstrap resamples of posts within each class,
seeded so they reproduce exactly. Paired intervals resample the same post IDs
for both models. They describe sampling uncertainty over posts only. They do
not include:

- **Run-to-run variation.** Requests set temperature 0, which reasoning
  models may not honour. On posts both attempts answered validly, the
  restarted Qwen run matched the stopped one on 174 of 176, but GPT-5 nano at
  low effort matched on only 64 of 103, and two label-only nano runs agreed on
  35 of 55. Differences near the edge of an interval could change on a rerun.
- **A finite-population correction.** The intervals treat the eligible posts
  as a sample from a broader population of comparable AITA posts. If the
  target were exactly the 1,839 eligible posts, a finite-population correction
  would narrow them; for the Sonnet-minus-Jev Brier difference, by roughly
  0.005 at each end. The conclusions do not change.
- **Multiple comparisons.** The paired table makes five comparisons with
  Jev. Sonnet's weighted Brier advantage has an interval that ends at −0.003,
  so it would not survive a Bonferroni correction. Its log-loss advantage
  (−0.149, interval −0.259 to −0.053) would.
- **Model version changes, prompt changes, and dataset contamination.**

The reliability plot and confidence gates use the probability assigned to the
chosen label for every model. Malformed answers have no probability, so they
are left out of the plot and its ECE but still count as wrong everywhere else.
The plot, ECE, and confidence gates are unweighted and describe the stratified
sample only. They are not population estimates, and weighting by class can
reorder models on ECE.
Jev's separate Choice confidence field is logged but not treated as the same
quantity as a class probability.

Malformed or missing chat probabilities remain failures and count against
top-1 accuracy and recall. Probability scores use valid distributions only and
show their coverage. Only Qwen produced malformed answers (4 of 770); scoring
them as the base-rate forecast instead leaves its weighted Brier at 0.410,
and so does scoring them as a uniform 25% forecast (0.4097 → 0.4100).

For future runs, a malformed answer will be scored as a uniform 25% forecast
in Brier and log loss, so a model can't gain by failing to answer. This rule is
declared here, before any such run; it is not applied to the final logs above.
Runs from now on also save each model's unparsed reply (`raw_output`), which
the final logs lack, so parser changes can be checked against what the model
actually said.

The consensus subset repeats sample accuracy and sample Brier on posts where
the official flair matches the vote-weighted four-verdict plurality. Selecting
that subset after seeing source comment data makes it a sensitivity check, not
a second independent test set.

## Latency and cost

Latency is wall time around the HTTP call, measured on one Apple M4 Pro laptop
on one evening. Each model's requests ran one at a time, but the models ran
side by side in separate processes. API calls went through OpenRouter, which
routed Jev to TypeSafe, Sonnet to Claude Platform on AWS and GPT-5 nano to
OpenAI, so they include network and routing time. Qwen and Gemma ran in LM
Studio on the same laptop, as 4-bit MLX builds, while the API runs were
going, after an untimed warm-up call. No request was retried. Treat the latency numbers as one
realistic client's view, not provider benchmarks.

Cost is the provider's billed amount per request, as reported by OpenRouter.
Jev bills input tokens only. Local models have no billed cost; their hardware
and electricity are not free and are not counted.

## Reproducing the sample

The full chain is checked by SHA-256 at every step:

1. The dataset's `dilemmas` table on Hugging Face. The final runs used the
   pinned `dilemmas/train.parquet` file (`40f43df2…`, revision `cb4c298c`),
   converted to JSONL by a pyarrow script that is now removed (see git
   history).
2. `python -m jevbench.sample_2025 --fetch` now downloads the same rows through
   the Hugging Face row API instead, with no dependencies, and checks them
   against `raw_sha256` (`aa9ca88f…`) in `data/final-ucb-2025.source.json`.
   The API output is byte-identical to the parquet export. The API serves
   the dataset's current version, so if the dataset changes, the check fails
   rather than silently building a different sample.
3. `python -m jevbench.sample_2025` selects the 770 posts with seed 20260923.
   The output hashes to `2f8cae7c…`, the `sample_sha256` in every final
   manifest.

The post text is not redistributed here; the dataset is CC BY 4.0 and the
steps above rebuild it.

## Earlier work

The 200-post development runs used the older `OsamaBsher/AITA-Reddit-Dataset`
sample. Their chat prompt showed option codes without the descriptions Jev
received, so their apparent Jev-versus-chat win was not a fair result. They
are not pooled with the final run. A separate Civil Comments calibration test
found Jev's expected calibration error (0.171) worse than GPT-5 nano's (0.125)
on 300 toxicity-rated comments; it does not bear on AITA probabilities. Both
sets of logs are in [`runs/dev-2023/`](../runs/dev-2023/), and the code that
produced them is at git tag `dev-2023`.

## Choosing Jev's question setup

The final run asks Jev one Choice question. Two rounds support that choice.

**Before the final run.** The 200-post development runs above compared the
direct question with 8 yes/no questions combined by a hand-written rule, and
with those 8 plus the direct question and a severity Score. The 8-question
version scored 41.5% against 53.0% for direct. A logistic regression on the
same 8 answers reached 51.5%, so most of the deficit came from the combining
rule rather than the questions. The 10-question version agreed with direct on 194 of 200 posts
(55.0% vs. 53.0%, McNemar p = 0.22).

**After the final run.** A check on 300 further 2023 posts tested four
redesigned setups, each one request per post
([`steps.py`](../jevbench/steps.py) has the exact wording):

- 5 yes/no questions: within their rights, harsh manner, other side
  unreasonable, clash of fair needs, poster downplaying their part.
- 2 severity Scores, one for each side, from "not wrong" to "caused real harm".
- 5 mixed steps: the two Scores, a Choice for what kind of thing the poster
  did, a Choice for what kind of conflict it is, and a Score for how one-sided
  the account is.
- The 5 mixed steps plus the direct question.

The direct question also ran on the same posts. Each setup's answers were
turned into four verdict probabilities by a multinomial logistic regression
(standardised features, L2 0.1, fixed step count), scored with 10-fold
cross-validation stratified by verdict, so no post was scored by a model fitted
on it. The direct answer went through the same regression, so every setup had
the same recalibration. Scores were weighted to the 2025 population mix, as in
the main results, and separately with all four verdicts weighted equally.

No setup clearly beat the direct question. Against the refitted direct answer, the
weighted Brier differences were −0.008 [−0.026, +0.010] for 5 yes/no, +0.009
for 2 Scores, +0.001 for 5 mixed steps and +0.002 with the verdict added. With
all four verdicts weighted equally, no difference excluded zero either; the
closest was −0.009 [−0.033, +0.016] for the steps plus the verdict. The extra
questions raised the billed cost by up to 1.5×. These comparisons are
exploratory: the regression uses one fixed L2 penalty and one fold assignment,
and the intervals resample the out-of-fold losses without refitting, so they
leave out that uncertainty. With no penalty, the 5 yes/no difference grows to
−0.013 [−0.039, +0.015], still inside the noise; with a strong one (L2 = 1),
both models collapse to always predicting NTA and the difference is zero.

The comparison is against the refitted direct answer, not the raw one. Every
step setup scores better than raw direct (0.354), but refitting direct alone
gives 0.320, so that gain is the regression's recalibration, not the extra
questions. The recalibration has a cost: fitted toward the population mix, every
refitted model, direct included, predicted ESH or NAH for almost no post (0–2%
recall), improving the weighted score by betting on NTA and YTA. Weighted
equally, the refitted models recover 51–64% recall on ESH and NAH. The refit
is not applied in the final comparison: it is post-processing, and the chat
models would need the same treatment.
Full tables: [`steps-summary.md`](../runs/diagnostic/steps-summary.md).

**Posts.** The 300 posts come from the older `OsamaBsher/AITA-Reddit-Dataset`,
chosen because only 6 ESH and 10 NAH posts from the 2025 source remain outside
the holdout. They share no IDs with the 2025 holdout, the 200-post
development sample or the pilot, and none had been sent to Jev before. They
were drawn, 35/35/15/15 per 100 posts, from a local 770-post file: an
800-post stratified sample made on 2026-09-22 with the `dev-2023` sampler and
seed 20260923, minus the 30 posts the current verdict-leak filter rejects
(that filter step is verified to reproduce the 770 exactly). The code for the
800-post draw was not committed at the time and rebuilding it was not
verified, so the pool cannot be re-drawn from this repository alone. The 300
post IDs and every answer are in the committed logs, which is what the tables
are computed from.
