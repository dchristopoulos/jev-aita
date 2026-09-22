# jev-bench

**Can a 200ms model judge you as well as Reddit can?**

[Jev](https://en.wikipedia.org/wiki/Jev_(AI_model)) is TypeSafe AI's "System One" model. It
doesn't generate text — it answers typed questions about a state and returns calibrated
probabilities. TypeSafe claims it's 40–200× faster and cheaper than frontier LLMs at this job. Measured here:
the speed holds, [the cost doesn't](#four-vendor-claims-checked).

r/AmItheAsshole is a near-perfect test for it. Every post is a moral judgment that a reader
makes in a couple of seconds, hundreds of thousands of people have already voted on the answer,
and the verdicts are a fixed set of four options that map exactly onto Jev's `Choice` primitive.

This repo measures whether Jev's probabilities are honest, whether it's actually cheaper, and
whether TypeSafe's own advice about *how* to ask makes it better.

## See it work

```bash
echo 'OPENROUTER_API_KEY=sk-or-v1-...' > .env    # gitignored; or export it
python -m jevbench.demo -n 3
```

You get the post, the model's reasoning, both verdicts and what Reddit actually said — so you
can read the case yourself and take a side:

```
aita for buying my younger cousins a year of disney+ for their christmas?
  r/AmItheAsshole · 94 points

  so i am an 18 year old girl who has 2 younger cousins ... i got them each a toy and a
  years subscription. i have since been told by my aunt and uncle this was a bad gift
  as they want to cut down on screen time ... i've since been told to keep it myself
  and get them new gifts

  The model's reasoning  (8 atomic questions, one call)
    █████████████·····  71%  op_disproportionate
    ████████████······  64%  op_disregarded
    ██████████········  58%  op_omits
    ███████···········  38%  other_behaved_badly
    ██················  12%  op_broke_agreement

  Verdicts
    Reddit said           NTA   Not the asshole — the other party is in the wrong
    Model, asked directly YTA   confidence 58%
    Composed in code      YTA   from the 8 answers above
    distribution: YTA 58%  ESH 22%  NTA 15%  NAH 5%

  Top comment on Reddit
    nta. aunt and uncle need to parent their children, and asking for more gifts is
    incredibly rude and entitled of them

  → disagrees with Reddit
```

`--verdict esh` to see only the hard ones, `--full-text` to read the whole post, `--model` to
put an LLM in the same seat.

## The hypothesis

TypeSafe's docs are explicit about how to use Jev well:

> "If the judgment you want depends on several independent factors, ask about each factor
> separately and combine the answers with your own logic."

"Who's the asshole here" is exactly such a judgment — it weighs what the poster did, what the
other party did, and whether either reaction was proportionate. So the same posts run three ways:

| Arm | Questions per call | Primitives | How the verdict is decided |
|---|---|---|---|
| `monolithic` | 1 | Choice | ask for the verdict directly |
| `decomposed` | 8 | Noul ×8 | eight atomic judgments, folded into a verdict **in code** |
| `full` | 10 | Noul ×8 + Choice + Score | both, plus a severity rubric |

No question in the `decomposed` arm mentions verdicts, assholes, or Reddit — there's a test
enforcing that. The model reports facts about the situation; the policy that turns those into a
judgment lives in [`compose_verdict`](jevbench/questions.py), where you can read it:

```python
op    = max(probability that the poster did something wrong)
other = max(probability that the other party did something wrong)

both wrong  -> ESH        poster only -> YTA
other only  -> NTA        neither     -> NAH
```

That's the argument for a System One model in nine lines: when you decide the threshold should
move, you edit a number instead of rewriting a prompt.

**The question nobody has measured: does following that advice actually help?**

## What's measured

| Metric | Question it answers |
|---|---|
| **Verdict accuracy** | Does it reach the same verdict as the crowd? |
| **vs. the majority verdict** | The margin over answering the most common verdict every time. On the stratified sample that constant scores 35%; on the real distribution it scores 78%, and [both are reported](#where-the-accuracy-number-stops-meaning-what-it-looks-like) because only one of them flatters the model. |
| **Confidence ECE** | When it says 80% sure, is it right 80% of the time? |
| **Reliability diagram** | *Where* is it miscalibrated? ECE alone hides this. |
| **Confusion table** | Which verdicts collapse into which. Watch for everything becoming NTA. |
| **p50 / p95 latency** | Tail latency, not the average — the tail is what users feel. |
| **$ / 1k posts** | Actual billed cost from real token counts. |
| **Fan-out ratio** | 1 → 8 → 10 questions. TypeSafe says extra questions are nearly free. |
| **Coverage @ confidence** | Auto-accept the sure calls, send the rest to a human. How much review can you actually eliminate? |
| **Unparseable** | How often an LLM emits a verdict that isn't one of the four. Scored as a distinct sentinel, never as one of the real verdicts — defaulting to the first option would let a failure score correct 35% of the time. Jev's constrained output makes this impossible, which is the structural argument for a typed model, measured rather than asserted. |

## Results

2,400 requests against 200 posts and 300 comments, **$0.2169** in billed cost, 0 failures on
Jev. (An earlier gpt-5-nano pass was discarded when every call came back truncated; its cost is
not in that figure because its records are not in `runs/`.)
Every number below comes out of [`runs/`](runs/), which is committed.

### The head-to-head

Same 200 posts, same labels, same code path.

| | Jev | gpt-5-nano (thinking off) | gpt-5-nano (effort=low) |
|---|---|---|---|
| verdict accuracy | **53.0%** | 36.5% | 45.0% |
| p50 latency | **0.38 s** | 1.27 s | 4.20 s |
| p95 latency | **0.53 s** | 2.49 s | 7.21 s |
| $ / 1k posts | **$0.035** | $0.043 | $0.158 |
| unparseable | **0** | 2 | 14 |

Jev wins all four at once. Whatever noise is in the labels handicaps both models
identically, so this comparison survives every objection below.

### Four vendor claims, checked

| Claim | Verdict | Measured |
|---|---|---|
| Typed output can't be malformed | **holds** | 0 unparseable in 1,200 Jev requests; nano 16 in 1,800 |
| Sub-second decisions | **holds** | p50 378 ms, p95 527 ms, flat across 1 → 10 questions |
| Extra questions are nearly free | **half** | latency 0.96×, but cost 1.67× — the questions ride in the prompt |
| 40–200× cheaper | **no** | 1.2× vs nano with thinking off ($0.035 vs $0.043) |

The fan-out result is a latency win, not a cost win. Jev's latency is flat at
0.96–0.98× going from 1 to 10 questions where nano's climbs to 1.26–1.57×, but
cost scales almost identically for both (Jev 1.42×/1.67×, nano 1.41×/1.72×).

### Where the accuracy number stops meaning what it looks like

The 53% is against a **stratified** sample whose majority baseline is 35%. On the
real r/AmItheAsshole distribution the story inverts. Reweighting the measured
per-class recalls by the true prior (78/15/5/2):

| | accuracy at the real prior |
|---|---|
| Jev, monolithic | 75.9% |
| Jev, full | 76.6% |
| **always answering NTA** | **78.0%** |

**Jev loses to a constant on the real distribution.** This README claims beating
that constant is the real bar, so it has to say plainly that it doesn't. The
`+18 pp` margin is a product of the stratification, not a property of the model.

The reason is visible in the confusion table: the four-way verdict collapses into
a YTA/NTA binary.

```
true\pred   yta   nta   esh   nah    recall
yta          40    28     1     1       57%
nta           6    60     1     3       86%
esh          12    17     1     0        3%
nah           6    19     0     5       17%
```

ESH recall is 1 in 30. A verdict the model essentially never predicts is a verdict
it cannot be trusted on, and averaging it into a single accuracy number hides that.

### The hypothesis was wrong

This README predicted, before the run, that the `decomposed` arm would fail because
`OP_FAULT` has four questions to `OTHER_FAULT`'s two, so a `max` gives the poster's
side twice as many chances to cross the threshold. The arm did fail — 41.5% against
53.0% — and it would have been easy to call that confirmation.

The data already on disk says otherwise. The probabilities are all in the run log,
so the check cost nothing:

| | fires at ≥ 0.5 |
|---|---|
| `max(OP_FAULT)` — four questions | 0.47 |
| `max(OTHER_FAULT)` — two questions | **0.69** |

The **two**-question side fires more often, not less. And the predicted direction is
backwards: the note said the fold tilts toward YTA, but the arm predicts YTA 24 times
against 70 true cases — it *under*-predicts it badly.

Balancing the counts doesn't rescue it either. The best 2-vs-2 subset reaches 46.0%
even when chosen with hindsight on the test set, and an oracle grid search over both
thresholds gets 45.0% — still nowhere near 53.0%.

What actually went wrong is the fold, not the counts. A 5-fold cross-validated
multinomial logistic regression on **the same eight nouls** scores 51.5%, within
noise of asking directly. The nouls carry the signal; `max()` with a 0.5 cutoff throws
it away by binarising the most informative question in the set. Decomposition didn't
fail here — my nine-line composition rule did.

### `full` does not beat `monolithic`

55.0% against 53.0% looks like a result and isn't. The two arms agree on 194 of 200
posts; of the 6 disagreements, `full` wins 5. McNemar's exact test gives **p = 0.22**.
The honest conclusion is that nine extra questions in the same request neither help
nor hurt the Choice they ride along with.

### Confidence gating is the strongest practical result

Jev's confidence does rank its own correctness, which is worth more than the accuracy
number. Acting only on answers above a threshold, with Wilson intervals because the
upper rows get thin fast:

| threshold | coverage | accuracy | 95% CI |
|---|---|---|---|
| ≥ 0.50 | 62.0% | 61.3% | [52.5, 69.4] |
| ≥ 0.70 | 39.5% | 68.4% | [57.5, 77.6] |
| ≥ 0.80 | 29.5% | 79.7% | [67.7, 88.0] |
| ≥ 0.90 | 16.0% | 90.6% | [75.8, 96.8] |
| ≥ 0.95 | 8.0% | 93.8% | [71.7, 98.9] |

The rise from 0.50 to 0.90 is real and is the shape you want for triage: auto-accept
the confident sixth, route the rest to a human. The 0.95 row is 16 posts and one
error — quoting "93.8%" from it would be false precision, and it's here only so the
thinning is visible.

### Calibration, measured against a probability instead of a coin flip

A Reddit verdict is binary, so it can't really test calibration: when the model says
0.7 there's no ground truth for how often 0.7 should be right. [Civil
Comments](https://huggingface.co/datasets/google/civil_comments) can, because its
`toxicity` field is *the fraction of human raters* who called a comment toxic. 0.6
means six of ten people said yes — which is exactly what a calibrated probability
should mean. 300 comments, stratified across five rater-agreement bands so the
contested middle isn't drowned by the ~71% of comments nobody disputes.

| | ECE ↓ | Brier ↓ | mean prediction | mean human fraction | Spearman ρ |
|---|---|---|---|---|---|
| Jev | 0.171 | 0.089 | 0.554 | 0.385 | **0.645** |
| gpt-5-nano | 0.125 | 0.078 | 0.485 | 0.385 | 0.606 |

Both **systematically over-predict** toxicity, Jev by +0.17 and nano by +0.10. Jev
looks worse, but the paired difference in Brier is **not** significant (95% CI
[-0.022, +0.003] on 291 shared comments), so "Jev is worse calibrated" is more than
the data supports.

The useful finding is the shape of the error. Jev *orders* comments better than nano
(ρ 0.645 vs 0.606) while sitting on the wrong scale — and a wrong scale with a right
ordering is a two-parameter problem. Fitting a slope and intercept on the log-odds on
a random half and scoring the held-out half:

| | ECE raw → calibrated | Brier raw → calibrated |
|---|---|---|
| Jev | 0.171 → **0.039** | 0.082 → **0.050** |
| gpt-5-nano | 0.124 → **0.027** | 0.074 → **0.054** |

Jev's ECE drops more than 4×, and after recalibration it has the better Brier score of
the two. **Out of the box its probabilities are not calibrated in the sense the word
usually means** — but they're ordered well enough that 150 labelled examples fix them.
If you plan to threshold on Jev's numbers, fit those two parameters on your own data
first.

Full tables: [`docs/results.md`](docs/results.md) and
[`docs/calibration.md`](docs/calibration.md).

### Reproducing it

```bash
./run.sh            # 200 posts, all arms, all models
./run.sh 50         # smaller and cheaper
```

`run.sh` prints the projected cost, judges two posts, and asks you to confirm before spending on
the rest. A **hard cap** (`BUDGET`, default `$1.00`) stops the run the instant it is reached —
whatever was measured is already on disk and reports fine. `--estimate` prices a run without
spending anything.

The default models are Jev and `gpt-5-nano`, which costs a few cents for 200 posts. A frontier
baseline costs roughly 50× the rest of the run combined, so it goes in a second pass once the
pipeline is proven, against the same cached sample:

```bash
python -m jevbench.bench -n 40 --models anthropic/claude-sonnet-5 --budget 0.75 \
    --out runs/frontier.jsonl
python -m jevbench.report runs/*.jsonl        # merged into one table
```

`report.py` takes several run logs and merges them, and warns if the models were not all scored
on the same posts — comparing a model judged on 200 posts against one judged on 40 different
ones is not a comparison.

It keeps three things, all meant to be committed:

| File | What it is |
|---|---|
| `runs/<timestamp>.jsonl` | one record per API request — probabilities, latency, tokens, cost |
| `docs/results.md` | the rendered tables |
| `docs/reliability.svg` | the calibration chart |

Together those make every number in this README reproducible by anyone who clones the repo — and
re-runnable against a later Jev version to see what changed. The steps run separately too:

```bash
python -m jevbench.bench -n 200 --out runs/mine.jsonl
python -m jevbench.report runs/mine.jsonl
```

Costs a few cents. `--arms`, `--models`, `--seed` to vary it. Records flush per request, so an
interrupted run keeps what it measured.

Reading posts and Reddit's verdicts needs no key at all — the data comes from HuggingFace:

```bash
python -m jevbench.demo --read -n 5
python -m jevbench.demo --read -n 3 --verdict esh --full-text
```

Tests need no API key either:

```bash
pytest
```

## Design notes

**Reddit is not a moral authority.** The target is what a particular crowd voted at a particular
time, and that crowd has known habits: it sides with the poster, it's harsher on mothers-in-law
than on posters, and posts are one-sided accounts by definition. A model that disagrees with
Reddit isn't necessarily wrong — which is exactly why `demo.py` shows you the post and lets you
judge for yourself. Treat "accuracy" as *agreement with r/AmItheAsshole*, nothing grander.

**The base rate is the whole trap.** NTA is ~78% of a random sample. A model that answers NTA
every time scores 78% and understands nothing. The sample is stratified to 35/35/15/15 and the
headline table reports the margin over always-answering-the-most-common-verdict, because the raw
accuracy number is close to meaningless on its own.

**The poster writes the evidence.** Every post is one side of a story, often omitting the part
that would change the verdict. That's a real limit on the ceiling here — and it's why
`op_omits` is one of the eight questions. Whether the model can detect a suspiciously one-sided
account is one of the more interesting things in the run.

**The fold is count-asymmetric — and that turned out not to be the problem.** `OP_FAULT` has
four questions to `OTHER_FAULT`'s two, so under a `max` the poster's side gets twice as many
chances to cross the threshold. That was the pre-registered explanation for why the decomposed
arm would underperform, written here before the run. The arm did underperform, and the
explanation was still wrong: the two-question side fires at 0.69 against the four-question
side's 0.47, and the predicted tilt toward YTA runs backwards in the data. The real cause is
that `max` with a 0.5 cutoff binarises away the probability mass that made the nouls worth
asking for — a logistic regression on the same eight nouls recovers almost all of it. Kept here
because a hypothesis that survived until the data arrived and then died is the most useful thing
in this repo; the full working is in [Results](#the-hypothesis-was-wrong).

**A composed verdict has no real confidence.** Noul returns a probability but no `confidence`
field — only Choice and Score carry one — so the decomposed arm synthesises confidence from how
far the *deciding* noul sat from 0.5. Context questions are excluded: letting `op_omits` land at
0.99 inflate the confidence behind a verdict it played no part in would corrupt exactly the
numbers the hypothesis rests on. There's a regression test for it.

**Score indexing is decided once per run.** TypeSafe's docs never pin whether Score levels start
at 0 or 1. Guessing per answer would scale some scores in a run differently from others, so
`raw_score` goes in the log and the report derives the convention once from the smallest score
it saw, and says which it used.

**Failed requests are recorded and excluded.** An API timeout is not a wrong verdict. Failures
are written to the run log, counted in the operations table, and kept out of accuracy — scoring
them would punish whichever model was flakiest rather than whichever was least accurate.

**ECE is not comparable across arms.** The monolithic and full arms report Jev's own Choice
confidence; the decomposed arm has no such field and synthesises one from how far the deciding
noul sat from 0.5. So "0.120 against 0.402" is partly real confidence against a proxy, and the
gap should not be read as a like-for-like calibration comparison.

**Every ECE is printed with its support.** ECE is a weighted mean over bins, and a bin holding
two samples moves it as if it were a measurement. The headline shows bins populated and the
smallest bin count so a reader can discount a number resting on almost nothing.

**Low-karma posts are dropped, and that filter is weaker than it sounds.** `min_score` filters
*post* karma, but the flair verdict is assigned from the top *comment*'s score, which the dataset
doesn't carry. So this is a proxy for how visible a post was, not for how many people voted on
its verdict. Accuracy does climb with post karma (45.5% → 62.7% across terciles, z = 2.00,
p = 0.045), but the honest reading is not "labels are noisier down there": the low tercile also
holds more ESH and NAH posts, and those are the classes the model already collapses. Within
YTA/NTA alone the gradient is 61.9% → 78.4%; within ESH/NAH it *reverses*. "Clearer cases are
easier" explains the pattern with fewer assumptions than "labels are noisier".

**The severity Score is directional only.** Reddit publishes a verdict, not a severity rating,
so the Score target is the verdict mapped onto the rubric rather than an independent human
judgment. Reported separately and labelled as such.

## Layout

```
jevbench/questions.py  the three arms, all three primitives, the composition rule
jevbench/metrics.py    calibration + cost math       (pure, fully tested)
jevbench/client.py     Jev + LLM behind one type     (retry, latency, cost, parse recovery)
jevbench/data.py       stratified r/AmItheAsshole sampler
jevbench/bench.py      runner, one JSONL record per request
jevbench/report.py     tables + reliability diagram
jevbench/civil.py      calibration against human rater fractions + recalibration
jevbench/demo.py       judge a few posts and show the working
runs/                  committed logs — every README number is auditable
```

## The API

Jev is reached through OpenRouter's alpha Decisions API, a different endpoint from chat
completions:

```python
POST https://openrouter.ai/api/alpha/decisions
{
  "model": "~typesafe/jev-latest",
  "state": "<the post>",
  "questions": {
    "op_disregarded": {
      "type": "noul",
      "instructions": "Did the poster override someone's clearly stated boundaries?",
      "criteria": {"true": "Someone said what they wanted and the poster went ahead anyway",
                   "false": "No boundary was stated, or the poster respected it"}
    },
    "verdict": {
      "type": "choice",
      "instructions": "Which verdict would the subreddit reach?",
      "criteria": {"yta": "...", "nta": "...", "esh": "...", "nah": "..."}
    }
  }
}
```

`noul` returns a probability, `choice` picks from options you define and returns a distribution
plus confidence, `score` places the state on an ordered rubric. Every question in a request is
evaluated in parallel against the same state.

## Data

[OsamaBsher/AITA-Reddit-Dataset](https://huggingface.co/datasets/OsamaBsher/AITA-Reddit-Dataset)
— 270k posts with the crowd's verdict and top comments, used for the verdict benchmark.

[google/civil_comments](https://huggingface.co/datasets/google/civil_comments) — 2M comments
whose labels are human rater *fractions* rather than binary calls, used for the calibration
arm. That difference is the whole reason it's here: a fraction can be compared against a
predicted probability directly, and a binary verdict cannot.

Both are fetched from the HuggingFace datasets-server with no authentication, cached to `data/`
and gitignored; they regenerate deterministically from `--seed`. Only post title and body, or
comment text, are ever sent to a model.

## License

MIT
