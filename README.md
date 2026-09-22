# Jev on Reddit AITA verdicts

**Bottom line:** Jev forecast second-best of seven setups, behind Sonnet 5,
at 1/60th of Sonnet's cost and 6× its speed. It was not the "40x-200x faster"
TypeSafe claims.

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is
TypeSafe's "System One" model. You ask it a multiple-choice question, and in
one fast call it returns a probability for each answer instead of writing text.
TypeSafe says it matches LLMs on quick judgments like this. I tested
`jev-1.13-20260917` against Sonnet 5, GPT-5 nano and two local open-weight
models on 770 posts from 2025 on Reddit's AITA forum, where readers vote on
who was at fault in a conflict.

Each model gives a probability for each of the forum's four verdicts:

| **YTA** | **NTA** | **ESH** | **NAH** |
|---|---|---|---|
| the poster is at fault | the other party is | everyone is | no one is |

I have no affiliation with TypeSafe and paid for every call myself (about
$2.70 in total).

## Results

![Weighted Brier score with 95% intervals for each model, against a no-model baseline](docs/headline.svg)

How to read the table:

- **Brier score** measures how good the four probabilities are, from 0
  (perfect) to 2. Lower is better.
- **Weighted** means scored as if the posts had the real forum's mix of
  verdicts (74% NTA). The sample deliberately includes more of the rare
  verdicts so they can be measured.
- **Top-1** is plain accuracy: how often the most likely verdict was right.
- **Macro recall** averages the hit rate of each of the four verdicts, so a
  model that always says NTA gets 25%.
- Brackets are 95% confidence intervals.

| Model | Weighted Brier ↓ [95% CI] | Weighted top-1 [95% CI] | Macro recall | Median call | $ per 1,000 posts |
|---|---:|---:|---:|---:|---:|
| Sonnet 5 | 0.344 [0.321, 0.370] | 76.9% [74.5, 79.1] | 36.1% | 2.46 s | $2.291 |
| **Jev, direct question** | 0.369 [0.344, 0.398] | 75.4% [72.7, 77.8] | 37.4% | 0.39 s | $0.037 |
| Qwen 3.6 35B-A3B, local | 0.410 [0.382, 0.437] | 75.3% [73.3, 77.2] | 30.3% | 1.62 s | not billed |
| GPT-5 nano, low effort | 0.480 [0.454, 0.509] | 66.1% [62.7, 69.4] | 35.5% | 4.83 s | $0.165 |
| Jev, two questions | 0.515 [0.497, 0.535] | 62.6% [59.6, 65.5] | 31.6% | 0.38 s | $0.037 |
| GPT-5 nano, minimal effort | 0.569 [0.552, 0.585] | 57.2% [53.7, 60.8] | 25.9% | 1.53 s | $0.056 |
| Gemma 4 26B-A4B, local | 0.588 [0.539, 0.636] | 58.3% [54.3, 62.3] | 44.4% | 1.57 s | not billed |
| *No model: base rates / always NTA* | *0.415* | *74.0%* | *25.0%* | | |

- **Sonnet 5 was best; Jev was a close second.** Sonnet's lead is 0.025
  (95% CI 0.003 to 0.046). Jev beat both GPT-5 nano settings and both local
  models.
- **Jev was fast and cheap, but not 40–200× faster.** Its median call took
  0.39 s, inside TypeSafe's 70–500 ms claim. That is 6.3× faster and 62×
  cheaper than Sonnet, and 3.9× faster and 1.5× cheaper than GPT-5 nano. The
  chat models here did little or no reasoning, while TypeSafe compared against
  a model with its default reasoning.
- **Guessing from base rates is hard to beat.** Only Sonnet and Jev clearly
  beat it. Always answering NTA is right 74% of the time, and every model
  misses more than half of the ESH and NAH posts.
- **Asking Jev two yes/no questions instead of one four-way question made it
  worse**, at least when the two answers are combined by multiplication.

Local models ran as 4-bit MLX builds on one laptop; Qwen's 4 malformed answers
count as wrong. More metrics, confusion matrices and calibration are in the
[full results](docs/results.md).

## What this does not show

Every model was asked for four probabilities: Jev through its native question
type, chat models through a prompt with the same verdict definitions that
returns JSON. Most people would ask a chat model for one label instead, so this
says nothing about label-only accuracy, or about speed and cost for a one-word
answer. It also says nothing about who is actually right: the target is the
forum's verdict, which matches the commenters' upvote-weighted verdict on 704
of 770 posts.

## How it was tested

- **Posts:** 770 from UC Berkeley D-Lab's
  [2025 dataset](https://huggingface.co/datasets/ucberkeley-dlab/fragility-moral-judgment-llms)
  (CC BY 4.0, pinned revision): 365 NTA, 285 YTA, 65 ESH and 55 NAH. Models
  saw only the title and text. Posts that state their own verdict were removed.
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

Details and limitations: [methodology](docs/METHODOLOGY.md).

## Reproduce

Python 3.10+, standard library only. Regenerate the results and charts from
the committed logs:

```bash
python -m jevbench.report runs/final-*.jsonl --priors data/final-ucb-2025.source.json --chart docs/headline.svg
```

To also get the flair-versus-comments check and reproduce
`docs/results.md` byte for byte, rebuild the source and sample (each is checked
against its SHA-256), then add `--source-raw data/ucb-2025-raw.jsonl`:

```bash
pip install pyarrow==25.0.1
curl -L -o data/ucb-2025-source.parquet https://huggingface.co/datasets/ucberkeley-dlab/fragility-moral-judgment-llms/resolve/cb4c298cbfa93ce9cdf56685f12a55b0a6928110/dilemmas/train.parquet
echo "40f43df275dfc56462453d5347f4927808adcf818095e5bcd436b35b945dbae7  data/ucb-2025-source.parquet" | shasum -a 256 -c
python scripts/export_source.py data/ucb-2025-source.parquet data/ucb-2025-raw.jsonl
python -m jevbench.sample_2025 --raw data/ucb-2025-raw.jsonl --out data/final-ucb-2025.jsonl
echo "2f8cae7c7bebe3c259efbe69664b463919a9db5b0db386d23a79d8892f82479d  data/final-ucb-2025.jsonl" | shasum -a 256 -c
```

New calls need an OpenRouter key in `.env` (see `.env.example`) and
`python -m jevbench.bench --help`. The runner has a spending cap and records
the exact prompts in each run's manifest.

```
jevbench/   runner, clients, metrics and report (stdlib only)
scripts/    source export and prompt verification (need pyarrow / tiktoken)
runs/       final request logs; diagnostic/ and dev-2023/ hold runs not used
docs/       results, methodology, protocol and charts
tests/      pytest suite, no network
```
