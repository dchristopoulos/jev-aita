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
them as the base-rate forecast instead leaves its weighted Brier at 0.410.

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
