# jev-bench

**I tested a new kind of AI model on Reddit's favourite question: am I the asshole?**

## Why

A new model called [Jev](https://en.wikipedia.org/wiki/Jev_(AI_model)) came out recently, and it
works differently from the chatbots we're used to. It doesn't write text. You give it a
situation and a few questions, and it answers each one with a probability: *yes, 82% sure*.
TypeSafe, the company behind it, calls it a "System One" model (fast, gut-feeling judgments)
and says it's **40–200× faster and cheaper** than a normal LLM at this kind of work.

That's a big claim, so I wanted to see for myself.

## What I did

I needed a test where the right answer is a quick judgment call, and
[r/AmItheAsshole](https://www.reddit.com/r/AmItheAsshole/) is perfect for that. Someone
describes a fight with their family, and thousands of strangers vote on who was wrong. Every
post ends in one of four verdicts:

| | |
|---|---|
| **YTA** | you're the asshole |
| **NTA** | not the asshole |
| **ESH** | everyone sucks here |
| **NAH** | no assholes here |

I took 200 real posts, showed each one to Jev and to OpenAI's cheapest model (`gpt-5-nano`),
and compared their verdicts with what Reddit decided. The models only ever saw the post itself,
never the votes or the comments.

Then I ran a second test on 300 comments from [Civil Comments](https://huggingface.co/datasets/google/civil_comments),
where groups of people rated whether each comment was toxic. That dataset tells you *what share*
of people said yes (say, 6 out of 10), so it can check whether Jev's "60% sure" really means 60%.

2,400 requests in total, and the whole thing cost **$0.22**.

## Results

### Jev vs gpt-5-nano, same 200 posts

| | Jev | gpt-5-nano (no thinking) | gpt-5-nano (thinking) |
|---|---|---|---|
| agrees with Reddit | **53.0%** | 36.5% | 45.0% |
| typical response time | **0.38 s** | 1.27 s | 4.20 s |
| slowest 5% | **0.53 s** | 2.49 s | 7.21 s |
| cost per 1,000 posts | **$0.035** | $0.043 | $0.158 |
| broken answers | **0** | 3 | 14 |

Jev was more accurate, faster and cheaper, all at the same time. Guessing at random would get
25%, and always answering the most common verdict gets 35% on this sample.

### Checking what TypeSafe claims

| Claim | Holds up? | What I measured |
|---|---|---|
| Much faster | ✅ yes | 3–11× faster, and it stays under half a second |
| Asking more questions is nearly free | ✅ for speed | 10 questions take the same time as 1, but cost 1.7× more |
| Answers are always well-formed | ✅ yes | 0 broken answers in 900 requests |
| 40–200× cheaper | ❌ no | about **1.2×** cheaper than the cheapest GPT |
| Its percentages are calibrated | ⚠️ not out of the box | see below |

## What I learned

**It's quick, and it knows when it's unsure.** This was the most useful result. When Jev said
it was at least 90% confident, it matched Reddit about **90%** of the time. That happened on about
1 post in 6. So you could let it handle the obvious cases on its own and send everything else to
a person.

**Its percentages run high, but they're easy to fix.** Against the toxicity ratings, Jev
consistently called comments more toxic than people did (GPT did too, a bit less). But it put
the comments in the right *order* a little better than GPT did. When the order is right and only the
scale is off, you can fix it with two numbers learned from 150 examples. On comments it
hadn't seen, that cut Jev's calibration error from 0.163 to **0.039**.

**Four verdicts shrank to two.** Jev handled YTA and NTA fine but almost never said "everyone
sucks here" (1 out of 30). Because most real AITA posts are NTA, answering "NTA" every time would
actually score slightly better than Jev on real Reddit (78% vs 76%). The 53% above is on a
sample I balanced on purpose so the rare verdicts had enough examples.

**My own idea didn't work, and I was wrong about why.** TypeSafe suggests splitting a big
question into small ones and combining the answers yourself. I tried that with 8 small
questions and a simple rule to combine them. It scored 41.5%, worse than just asking directly.
Before running it I'd written down why I thought it would fail, and when I checked, the data
said my explanation was wrong. The small questions had the right information. My 9-line rule
for combining them threw most of it away.

**The cost claim didn't hold up.** 40–200× might be true against big models, but I only tested
a small, cheap one, and against that it's about 1.2×. It's still cheaper, just not by the amount
in the headline.

## Try it

You need Python 3.10 or newer. You can browse the posts and Reddit's verdicts without any
API key:

```bash
git clone https://github.com/dchristopoulos/jev-bench && cd jev-bench
python -m jevbench.demo --read -n 3
```

To watch Jev judge them, add an [OpenRouter](https://openrouter.ai) key (a few cents is plenty):

```bash
cp .env.example .env        # paste your key in
python -m jevbench.demo -n 3
```

You'll see the post, Jev's reasoning, its verdict next to Reddit's, and Reddit's top comment,
so you can decide who you agree with. This is a real answer from the benchmark log: Jev agrees
with Reddit here but isn't sure, and says so.

```
aita for buying my younger cousins a year of disney+ for their christmas?

  The model's reasoning
    ████████████████··  90%  other_escalated
    ██████████████····  80%  stakes_high
    █████████████·····  73%  op_omits
    ████████████······  68%  other_behaved_badly
    ██████████········  56%  op_disregarded

  Reddit said   NTA
  Jev said      NTA   confidence 29%
                      NTA 46% · YTA 32% · ESH 20% · NAH 2%

  Top comment on Reddit
    nta. aunt and uncle need to parent their children, and asking
    for more gifts is incredibly rude and entitled of them
```

To rerun the whole benchmark: `./run.sh` (it shows the cost and asks before spending anything).

## How it's built

- **Python, standard library only.** No dependencies to install, just `urllib` talking to
  OpenRouter.
- **Every request is saved.** All 2,400 are in [`runs/`](runs/) with their answers, timing
  and cost, so you can check any number here yourself.
- **A spending cap.** The runner stops the moment it reaches the budget you set.
- **77 tests** covering the maths, the API parsing and the data sampling. They run without a
  key.

The full write-up has the question design, every table, confidence intervals, the statistics,
and the bugs I hit along the way: **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**.

## Data

- [AITA Reddit dataset](https://huggingface.co/datasets/OsamaBsher/AITA-Reddit-Dataset): 270k posts with Reddit's verdict
- [Civil Comments](https://huggingface.co/datasets/google/civil_comments): 2M comments with human toxicity ratings

Both download free from HuggingFace, no account needed.
