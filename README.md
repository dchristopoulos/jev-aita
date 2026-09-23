# Jev vs. LLMs on Reddit's "Am I the Asshole?"

**Bottom line:** In the original seven-setup benchmark, Jev came second,
behind Sonnet 5 (Brier score 0.369 vs. 0.344; lower is better). Jev's median
call was 6.3× faster than Sonnet's, and 62× cheaper. That's fast, but not
the "40x-200x faster" TypeSafe claims.

![Weighted Brier scores with 95% intervals for the original runs and two later adjusted Jev results](docs/headline.svg)

The chart includes two later Jev follow-ups. I did not retrain Jev itself:
a small logistic regression learned from its answers to 300 labeled 2023
posts, then adjusted its probabilities on the 770 test posts. That gives Jev
the lowest score shown (0.337), but Sonnet's 0.344 is unadjusted, so it is
not a like-for-like win. A separate analysis gave every model the same kind
of adjustment; Sonnet still led by 0.007, an inconclusive gap.

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is
TypeSafe's "System One" model. It doesn't write text: you give it a situation
and a question, and in one fast call it returns a probability for each answer.
TypeSafe says it matches LLMs on quick judgments like this. I tested
`jev-1.13-20260917` on posts from Reddit's r/AmItheAsshole, where readers vote
on who was at fault in a conflict:

| **YTA** | **NTA** | **ESH** | **NAH** |
|---|---|---|---|
| the poster is at fault | the other party is | everyone is | no one is |

The test had two parts. First, Jev alone: find the best way to ask it.
Second, that setup against Sonnet 5, GPT-5 nano and two local open-weight
models, on 770 posts from 2025. Every model gives a probability for each of the
four verdicts, and is scored against Reddit's verdict.

I have no affiliation with TypeSafe and paid for every call myself (about
$2.75 in total).

## What counts as "correct"

Whether someone is the asshole is an opinion, so there is no true answer
here. This benchmark measures something narrower: **can a model predict the
verdict Reddit gave?** The target is the post's official verdict flair, which
AITA sets from the top-voted comment. A model is "right" when its top verdict
matches that flair. The main score, the Brier score, also gives partial
credit: putting 70% on Reddit's verdict scores better than putting 30% on it.

Reddit's verdict is itself noisy, because one top comment decides it even when
other commenters disagree. On 704 of the 770 posts, the flair matches what most
commenters said (weighted by upvotes); on the other 66, it doesn't. Scoring
only the 704 clearer posts tells the same story: Jev 63.4% vs. Sonnet 64.3%
plain accuracy ([details](docs/results.md#flair-and-comment-disagreement)).
"Not enough info" posts are left out, since that's not a verdict on anyone.

## Part 1: finding the best way to ask Jev

In one call, Jev can answer one multiple-choice question (pick one of the four
verdicts) or several yes/no and rating questions. We tried eight ways of asking.
None was clearly better than simply asking directly: "which verdict would the
subreddit reach?" So the main benchmark uses that direct question.

**First round: 200 posts, before the benchmark.** Asking directly got 53.0%
right. Asking eight yes/no questions about the facts, then turning the answers
into a verdict, got 41.5% with a hand-written rule and 51.5% with a rule
learned from data. Ten questions, including the direct one, got 55.0%: four more posts right
than direct, which could easily be luck (p = 0.22).

**Second round: 300 unused 2023 posts, after the benchmark.** This time every
setup, direct included, got the same *trained fix*: a small regression that
turns Jev's answers into four verdict probabilities. For direct, it
corrects Jev's probabilities; for the yes/no setups, it is also what turns the
yes/no answers into a verdict. Each post was scored by a
fix trained on the other posts, never on itself. Because of the fix, these
numbers can't be compared with Part 2, which scores Jev's raw answers.
Lower Brier is better; a negative difference means the setup beat direct.
Accuracy is weighted to Reddit's real verdict mix (explained in Part 2).

| Setup | Questions per call | Weighted Brier ↓ | Difference from direct | Weighted accuracy | Median call | $ per 1,000 posts |
|---|---:|---:|---:|---:|---:|---:|
| Direct, with fix | 1 | 0.320 | baseline | 79.3% | 525 ms | $0.034 |
| Five yes/no | 5 | 0.312 | −0.008 | 80.6% | 529 ms | $0.040 |
| Two severity scores | 2 | 0.329 | +0.009 | 80.0% | 526 ms | $0.035 |
| Five mixed steps | 5 | 0.322 | +0.001 | 80.9% | 531 ms | $0.046 |
| Five steps plus direct | 6 | 0.322 | +0.002 | 79.8% | 520 ms | $0.052 |

- Five yes/no looked best (0.008 better than direct), but that could be
  noise: the 95% confidence interval (CI) runs from 0.026 better to 0.010
  worse.
- Extra questions barely slowed calls down, but cost more.
- Most of the gain came from the fix, not from extra questions: direct alone
  went from 0.354 raw to 0.320 with the fix.
- The fix almost never picked ESH or NAH (it caught 0–2% of them), because it
  learned to favor the common verdicts.

The benchmark also ran a two-yes/no setup, which multiplies two answers into
four verdict probabilities. It did much worse than direct (0.515 vs. 0.369).
The [full development tables](runs/diagnostic/steps-summary.md),
[later question wording](jevbench/steps.py), and [two-question wording](jevbench/questions.py)
give the remaining detail.

## Part 2: Jev vs. LLMs on 770 posts

How to read the table:

- **Brier score** grades all four probabilities, from 0 (perfect) to 2
  (worst). Lower is better. Being confidently wrong costs more than hedging.
- **Weighted:** on Reddit, 74% of verdicts are NTA and about 4% each are ESH
  and NAH. The sample has about twice as many ESH and NAH posts as that, so
  there are enough of them to measure. Weighting puts each verdict back at its
  real share: an NTA post counts about 1.6× as much as in a plain average, an
  ESH post about half as much.
- **Top-1:** how often the model's top verdict matched Reddit's, weighted the
  same way.
- **Macro recall:** how often each of the four verdicts was caught, averaged.
  A model that always says NTA gets 25%. This rewards catching the rare
  verdicts.
- **Brackets** are 95% confidence intervals (CI), from resampling posts. They show how much a score
  depends on which posts were picked, not how much a model varies between runs:
  each model ran once, and even at temperature 0 a restarted GPT-5 nano run
  repeated only 64 of 103 answers.

| Model | Weighted Brier ↓ [95% CI] | Weighted top-1 [95% CI] | Macro recall | Median call | $ per 1,000 posts |
|---|---:|---:|---:|---:|---:|
| Sonnet 5 | 0.344 [0.321, 0.370] | 76.9% [74.5, 79.1] | 36.1% | 2.46 s | $2.291 |
| **Jev, direct question** | 0.369 [0.344, 0.398] | 75.4% [72.7, 77.8] | 37.4% | 0.39 s | $0.037 |
| Qwen 3.6 35B-A3B, local | 0.410 [0.382, 0.437] | 75.3% [73.3, 77.2] | 30.3% | 1.62 s | not billed |
| *No model: base rates / always NTA* | *0.415* | *74.0%* | *25.0%* | | |
| GPT-5 nano, low effort | 0.480 [0.454, 0.509] | 66.1% [62.7, 69.4] | 35.5% | 4.83 s | $0.165 |
| Jev, two questions | 0.515 [0.497, 0.535] | 62.6% [59.6, 65.5] | 31.6% | 0.38 s | $0.037 |
| GPT-5 nano, minimal effort | 0.569 [0.552, 0.585] | 57.2% [53.7, 60.8] | 25.9% | 1.53 s | $0.056 |
| Gemma 4 26B-A4B, local | 0.588 [0.539, 0.636] | 58.3% [54.3, 62.3] | 44.4% | 1.57 s | not billed |

- **Sonnet 5 came first, Jev second.** Sonnet's lead is 0.025 (95% CI 0.003 to
  0.046). That is outside noise on its own, but not after correcting for
  comparing Jev with five models at once (Bonferroni), so call it borderline.
  On another score, log loss, Sonnet's lead is clearer and survives that
  correction. Without weighting, there's no clear difference (Jev 0.556, Sonnet 0.563).
  Jev beat both GPT-5 nano settings and both local models.
- **Jev was 6.3× faster than Sonnet, not 40–200×.** Its median call took 0.39 s,
  inside TypeSafe's 70–500 ms claim, and it was 62× cheaper than Sonnet.
  Against GPT-5 nano it was 3.9× faster and 1.5× cheaper than minimal effort,
  and 12× faster and 4.5× cheaper than low effort. TypeSafe compared against a
  model reasoning at its default level; the chat models here did little or no
  reasoning, which makes them much faster.
- **Guessing from base rates is hard to beat.** Always saying NTA is right 74%
  of the time. The "no model" row gives every post Reddit's base rates. Only
  Sonnet and Jev clearly beat it (by 0.070 and 0.046). Qwen was slightly
  better on paper (0.410 vs. 0.415), but that could be noise (95% CI: from
  0.033 better to 0.022 worse than base rates). Every model missed more than half of the ESH and NAH posts.

Local models ran as 4-bit MLX builds on one laptop; Qwen's 4 malformed answers
count as wrong. More metrics, confusion matrices and calibration are in the
[full results](docs/results.md).

### Follow-ups: adding a trained fix

A model's probabilities are often off in predictable ways, for example too
confident. A common fix is to train a small regression on answers where the
right verdict is known, then use it to correct new answers. The model itself
doesn't change. The main table above uses no fix. These two follow-ups test
what a fix does.

**1. Jev with a fix trained on 2023 posts.** We trained the fix on Jev's
answers to 300 older posts from 2023, then applied it to Jev's answers on the
770 test posts. Only Jev got this fix, so don't compare these numbers with
Sonnet's in the main table; follow-up 2 does that fairly.

| Jev setup | Weighted Brier ↓ [95% CI] | Weighted top-1 [95% CI] | Macro recall |
|---|---:|---:|---:|
| One question, no fix (from the table above) | 0.369 [0.344, 0.398] | 75.4% [72.7, 77.8] | 37.4% |
| One question, with fix | 0.337 [0.321, 0.354] | 78.6% [77.0, 80.1] | 33.0% |
| Five yes/no questions, with fix | 0.346 [0.332, 0.361] | 76.4% [74.8, 77.8] | 30.1% |

- The fix improved Jev's Brier score (0.369 → 0.337), but it never picks ESH or
  NAH as the answer.
- Five questions did not beat one question: 0.009 worse, which could be noise
  (95% CI: from 0.001 better to 0.019 worse). They cost a little more ($0.044 vs. $0.037 per
  1,000 posts) and were a little slower (0.57 s vs. 0.39 s per call, measured
  on a different day).
- We picked this follow-up after seeing the test results, so treat it as
  exploratory, not as a clean test.

[Plan, full results and run log](docs/FIVE_QUESTION_FOLLOWUP.md).

**2. Every model with the same fix.** For a fair comparison, every model got
the same kind of fix, applied to its saved answers (no new calls). This fix was
trained on the 770 test posts themselves, but no post was corrected by a fix
that had seen it: the posts were split into 10 groups, and each group was
corrected by a fix trained on the other nine. The method was written down
and committed before this was run, though after the main results were known.

- Sonnet's lead over Jev shrank from 0.025 to 0.007 (Jev 0.339 with the fix),
  small enough to be noise: the 95% CI runs from Sonnet ahead by 0.021 to Jev
  ahead by 0.007.
- The fix helped every model and bunched them together: Gemma went from last
  (0.588) to 0.352, and GPT-5 nano minimal ended at 0.410, about the base-rate
  score (0.415).
- The fix can also change which verdict a model picks, so this is not purely
  a confidence correction.
- Both models got better on NTA posts and worse on YTA, ESH and NAH posts.
  After the fix, no model picked ESH or NAH for any post.

[Plan and results](docs/RECALIBRATION_PLAN.md).

## What this does not show

- **Who is actually right.** Only how well each model predicts Reddit's verdict
  ([see above](#what-counts-as-correct)).
- **Label-only use.** Every model had to give four probabilities, Jev natively
  and the chat models as JSON. Asking a chat model for one label would change
  its accuracy, speed and cost.
- **Full-size local models.** Qwen and Gemma ran as 4-bit builds with reasoning
  off. The prompts also differ slightly: Jev is asked which verdict the
  subreddit would reach, the chat models to predict the official flair.

## Method details

- **Posts:** 770 from UC Berkeley D-Lab's
  [2025 dataset](https://huggingface.co/datasets/ucberkeley-dlab/fragility-moral-judgment-llms)
  (CC BY 4.0, pinned revision): 365 NTA, 285 YTA, 65 ESH and 55 NAH. Models
  saw only the title and text. Posts that state their own verdict ("EDIT:
  seems I'm NTA") were removed. None of the Part 1 posts are among them.
- **Plan:** the [protocol](docs/FINAL_PROTOCOL.md) lists four deviations from
  it, including a parser fix that restarted two runs. Neither the plan nor the
  exact code that ran was committed before the runs, so their timing can't be
  proven from git; the protocol lists what can still be checked.
- **Prompts:** in [`client.py`](jevbench/client.py) and
  [`questions.py`](jevbench/questions.py), pinned by a test and checked against
  billed token counts by [a script](scripts/check_prompt_tokens.py).
- **Intervals:** 95% bootstrap over posts within each verdict. They exclude
  run-to-run variation: at temperature 0, a restarted GPT-5 nano run repeated
  only 64 of 103 answers.
- **Latency:** one laptop in UTC+3, via OpenRouter, one evening. Jev is served
  from the US West Coast, so its times include a long network trip.

More: [methodology](docs/METHODOLOGY.md), [run logs](runs/README.md).

## Try it

Python 3.10+, standard library only. Clone the repo and run from its folder.

Anything that asks Jev live needs an **OpenRouter API key**, because this repo
calls Jev through OpenRouter. Create one at
[openrouter.ai/keys](https://openrouter.ai/keys) and add a little credit; a
dollar covers thousands of calls. [OpenRouter's terms](https://openrouter.ai/terms)
currently set a $5 minimum credit purchase, in US dollars, if you need to top
up. That is prepaid credit, not the cost of one run. Then copy `.env.example`
to `.env` and paste the key in. Rebuilding the results from the logs needs no
key.

### Judge a real AITA post

This pulls a random real post, with Reddit's verdict, from the same Hugging
Face dataset, and asks Jev live (two calls, about $0.0001). It shows both
setups for comparison: Step 1 is the direct question used in the benchmark;
Step 2 is the two-yes/no setup, which scored worse across the benchmark.

```bash
python -m jevbench.show --random
```

One live run, for illustration (not saved; numbers vary between calls):

```
━━ POST 1ilu5ya ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3071 chars
  AITA for telling my boyfriend not to bother coming over to my house anymore?
  …
  Forum verdict: NTA (Not the asshole -- the other party is in the wrong)

━━ STEP 1 · Direct question ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 553 ms · $0.000047
  Answered: choice NTA, confidence 0.71

  NTA   0.78  ███████████████████████████████  ← picked · forum verdict
  YTA   0.14  ██████
  NAH   0.04  ██
  ESH   0.04  ██

  Jev picked NTA · forum said NTA · ✓ right · Brier 0.071

━━ STEP 2 · Two yes/no questions ━━━━━━━━━━━━━━━━━━━━━━━━━━━ …
```

The post may be one of the 770 benchmark posts; its answer is fetched fresh
either way. To judge your own post instead, pass the text (first line is the
title), or pipe it in with `--text -`:

```bash
python -m jevbench.show --text "AITA for not lending my car to my sister?
She crashed my last one and never paid for the repairs."
```

### Rebuild the results

No key or network needed. This regenerates every table and chart from the
committed logs:

```bash
python -m jevbench.report runs/final-*.jsonl \
  --priors data/final-ucb-2025.source.json \
  --chart docs/headline.svg --five-followup runs/followup-yesno5-2025.jsonl
```

It rewrites `docs/results.md` without one section (Reddit's verdict against
the comment vote), which needs the posts from the next step. The Part 1 tables
rebuild the same way:

```bash
python -m jevbench.steps fit runs/diagnostic/steps-dev-*.jsonl
```

And follow-up 2, the same fix for every model (about a minute, no key):

```bash
python -m jevbench.recalibrate
```

### Replay a benchmark post

To see a post from the benchmark next to Jev's logged answer and Reddit's
verdict, first rebuild the 770 posts. This downloads the dataset from the same
Hugging Face API `--random` uses, refuses it if it doesn't
match the pinned hash, and draws the same sample:

```bash
python -m jevbench.sample_2025 --fetch --raw data/ucb-2025-raw.jsonl --out data/final-ucb-2025.jsonl
echo "2f8cae7c7bebe3c259efbe69664b463919a9db5b0db386d23a79d8892f82479d  data/final-ucb-2025.jsonl" | shasum -a 256 -c
```

Then:

```bash
python -m jevbench.show 1hvkncw
```

Leave out the id for the first post, add `--live` to ask Jev again, or
`--short` to print only the post's first lines (this works with `--random` and
`--text` too). With the posts rebuilt, adding
`--source-raw data/ucb-2025-raw.jsonl` to the report command restores its
missing section.

### Run a new benchmark

`python -m jevbench.bench --help`. The runner has a spending cap, and each
run's manifest records the exact prompts.

```
jevbench/   runner, clients, metrics, report, viewer, step setups, recalibration (stdlib only)
scripts/    prompt verification against billed tokens (needs tiktoken)
runs/       final logs; diagnostic/ holds Part 1 and other runs; dev-2023/ the first round
docs/       results, methodology, protocol and charts
tests/      pytest suite, no network
```
