# jev-bench

**I tested a new kind of AI model on Reddit's favourite question: am I the asshole?**

## Why

A new model called [Jev](https://en.wikipedia.org/wiki/Jev_(AI_model)) came out recently, and it
works differently from the chatbots we're used to. It doesn't write text. You give it a
situation and a few questions, and it answers each one with a probability: *yes, 82% sure*.
TypeSafe, the company behind it, calls it a "System One" model (fast, gut-feeling judgments)
and says it's **40–200× faster and cheaper** than a normal LLM at this kind of work.

That's a big claim, so I wanted to see for myself. I gave myself a $3 budget and picked
r/AmItheAsshole because I wanted a test that's actually fun to read through.

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

I took 200 real posts and, through [OpenRouter](https://openrouter.ai), showed each one to Jev and to OpenAI's cheapest model (`gpt-5-nano`),
and compared their verdicts with what Reddit decided. The models only ever saw the post itself,
never the votes or the comments.

Then I ran a second test on 300 comments from [Civil Comments](https://huggingface.co/datasets/google/civil_comments),
where groups of people rated whether each comment was toxic. That dataset tells you *what share*
of people said yes (say, 6 out of 10), so it can check whether Jev's "60% sure" really means 60%.

2,400 requests in total, and the whole thing cost **$0.22**.

## How Jev works

Talking to Jev is different from chatting with GPT. There's no conversation. Each request has two
parts: the **situation** (here, the Reddit post) and a list of **questions** about it. Jev
answers every question at the same time and sends back numbers instead of sentences.

There are three kinds of question:

| Type | You ask | You get back |
|---|---|---|
| **Noul** | a yes/no question | how likely "yes" is, e.g. `0.74` |
| **Choice** | pick one option from a list | the pick, the odds for each option, and how confident it is |
| **Score** | where this falls on a scale you describe | a position on the scale, plus confidence |

For each post I asked 10 questions in one request, using all three types:

| Question | Type | In plain words |
|---|---|---|
| `op_broke_agreement` | noul | Did the poster break a promise? |
| `op_disproportionate` | noul | Did the poster overreact? |
| `op_disregarded` | noul | Did the poster ignore someone's clearly stated wishes? |
| `op_deceived` | noul | Did the poster lie or hide something? |
| `other_behaved_badly` | noul | Did the other person behave badly? |
| `other_escalated` | noul | Did the other person make it worse? |
| `stakes_high` | noul | Is something serious at stake (money, housing, a relationship)? |
| `op_omits` | noul | Is the poster probably leaving out details that make them look bad? |
| `verdict` | choice | YTA, NTA, ESH or NAH? |
| `severity` | score | How wrong was the poster, from "blameless" to "caused real harm"? |

Each question also comes with a short description of what "yes" and "no" mean, so Jev knows
where the line is. None of the 8 small questions mention Reddit or verdicts, and a test checks
that. They only ask about what happened.

That gave me two ways to reach a verdict from the same request:

1. **Ask directly**: just read the answer to the `verdict` question.
2. **Work it out myself**: take the 8 yes/no answers and apply a simple rule. If the poster did
   something wrong and the other person didn't, it's YTA. If only the other person did, NTA.
   Both, ESH. Neither, NAH.

TypeSafe's docs recommend the second approach. Testing whether that advice actually helps was
one of the main reasons I built this.

## What it looks like

A real answer from the benchmark, and one of my favourites because Jev disagrees with Reddit
and you can see why:

```
aita for standing my ground?
  r/AmItheAsshole · 11 points

  i (26f) have been with my (27m) partner for almost 3 years ... he wishes to name a son
  first name his grandfathers name middle name his name. i think it is sweet he wants to
  honor the family names but i hate that idea ...

  The model's reasoning  (8 yes/no questions, one call)
    █████████████·····  74%  stakes_high
    ████████··········  47%  op_omits
    ██████············  35%  op_disregarded
    ████··············  21%  op_disproportionate
    ███···············  18%  other_escalated
    ███···············  16%  other_behaved_badly
    █·················   7%  op_broke_agreement
    █·················   6%  op_deceived

    severity 8% (confidence 76%)

  Verdicts
    Reddit said           NTA
    Asked directly        NAH   confidence 32%
    Worked out in code    NAH   from the 8 answers above
    distribution: NAH 48%  NTA 47%  YTA 4%  ESH 1%

  Top comment on Reddit
    nta. naming a child is for both parents to do and is a one vote no / two votes
    yes system. he cannot unilaterally pick the names ...

  → disagrees with Reddit
```

Reddit sided with the poster. Jev says **nobody** is the asshole, and you can follow its
reasoning. It thinks the stakes are high (74%), but it doesn't think the boyfriend behaved
badly (16%) or that the poster did (every "did the poster…" answer is under 50%). No baby exists
yet, and he only *wants* a name. That's a disagreement, not bad behaviour.

It also told us it wasn't sure: 48% NAH against 47% NTA, with a confidence of 32%. That matters.
In a real system this is the kind of answer you'd pass to a person instead of trusting it.

The whole thing took **0.43 seconds** and cost a fraction of a cent.

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

## Can it do better?

Yes, very likely. These are the results for one setup, and most of it can be tuned:

- **The rule that combines the 8 answers.** Mine was simple and lost information. A small trained model on the same answers already got 51.5% instead of 41.5%.
- **The question wording.** Small changes to how a question is phrased change the answers.
- **Calibration.** Two numbers fitted on 150 examples already fixed most of the over-confidence.
- **The confidence cut-off.** Where you set it decides how much Jev handles alone and how accurate that part is.

So read the numbers as a starting point, not a ceiling.

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

You'll see output like the example above for each post.

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
