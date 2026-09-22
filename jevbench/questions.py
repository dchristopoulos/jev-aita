"""The question sets under test.

TypeSafe's docs say to decompose a judgment that weighs several independent
factors into atomic questions and combine them in code. "Who is the asshole
here" is exactly such a judgment -- it weighs what the poster did, what the
other party did, and whether either reaction was proportionate.

So we run the same posts three ways:

  monolithic  one Choice: "what is the verdict?"
  decomposed  eight atomic nouls, folded into a verdict by code below
  full        all of the above plus a Score, exercising all three primitives

Reddit's own verdict is the target. It is a crowd opinion, not a moral fact --
see the README.
"""

from __future__ import annotations

from .data import VERDICTS

# Atomic gut-checks. Each is the kind of judgment a reader makes in a second,
# which is what a System One model is built for. Split into the two sides of
# the conflict so code can tell YTA from ESH from NAH.
OP_FAULT = {
    "op_broke_agreement": (
        "Did the poster break a promise, agreement, or clearly understood rule?",
        "The poster went back on something they had committed to",
        "No commitment was broken, or none existed",
    ),
    "op_disproportionate": (
        "Was the poster's reaction out of proportion to what happened?",
        "The response was far harsher or larger than the situation called for",
        "The response was proportionate, or restrained",
    ),
    "op_disregarded": (
        "Did the poster override someone's clearly stated feelings, needs, or boundaries?",
        "Someone said what they wanted and the poster went ahead anyway",
        "No boundary was stated, or the poster respected it",
    ),
    "op_deceived": (
        "Did the poster lie, hide something, or mislead anyone involved?",
        "Deliberate deception or a meaningful omission by the poster",
        "The poster was straightforward with everyone involved",
    ),
}

OTHER_FAULT = {
    "other_behaved_badly": (
        "Did the other person in this story behave badly?",
        "The other party acted unfairly, cruelly, or dishonestly",
        "The other party behaved reasonably throughout",
    ),
    "other_escalated": (
        "Did the other person escalate the conflict beyond what the poster did?",
        "The other party raised the stakes: insults, ultimatums, involving others",
        "The other party kept it proportionate, or de-escalated",
    ),
}

CONTEXT = {
    "stakes_high": (
        "Does this conflict have meaningful real-world consequences?",
        "Money, housing, health, a relationship, or a job is genuinely at stake",
        "A minor disagreement with little lasting consequence",
    ),
    "op_omits": (
        "Is the poster likely leaving out details that would make them look worse?",
        "The account is one-sided, vague about the poster's own conduct, "
        "or reports reactions far stronger than the events described would explain",
        "The account reads as complete and even-handed about the poster's own part",
    ),
}

DIMENSIONS = {**OP_FAULT, **OTHER_FAULT, **CONTEXT}

# The questions compose_verdict actually reads. CONTEXT is deliberately excluded:
# stakes_high and op_omits inform a human reader but must not move the verdict --
# and must not move the confidence derived from it either.
DECIDING = {**OP_FAULT, **OTHER_FAULT}

SEVERITY_LEVELS = [
    "Blameless: the poster did nothing wrong",
    "Minor: a small lapse most people would forgive",
    "Clearly wrong: the poster owes an apology",
    "Seriously wrong: the poster caused real harm and should make amends",
]

VERDICT_QUESTION = (
    "You are judging a post from r/AmItheAsshole. "
    "Which verdict would the subreddit reach about the poster?"
)


def _noul(instructions: str, yes: str, no: str) -> dict:
    return {"type": "noul", "instructions": instructions,
            "criteria": {"true": yes, "false": no}}


def _choice(instructions: str, criteria: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def monolithic() -> dict[str, dict]:
    """Ask for the verdict directly -- the naive framing, and the control."""
    return {"verdict": _choice(VERDICT_QUESTION, dict(VERDICTS))}


def decomposed() -> dict[str, dict]:
    """Eight atomic nouls. No question mentions the verdict; code derives it."""
    return {name: _noul(*spec) for name, spec in DIMENSIONS.items()}


def full() -> dict[str, dict]:
    """All three primitives in one request: 8 nouls, 1 choice, 1 score.

    Ten questions, one state, evaluated in parallel. If Speculative Fan-Out
    holds, this costs barely more than the single-question arm.
    """
    qs = decomposed()
    qs["verdict"] = _choice(VERDICT_QUESTION, dict(VERDICTS))
    qs["severity"] = _score("How badly did the poster behave?", SEVERITY_LEVELS)
    return qs


# --- Composing atomic answers into a verdict ------------------------------

def compose_verdict(probs: dict[str, float], at: float = 0.5) -> str:
    """Fold the atomic judgments into one of the four verdicts.

    This is the step TypeSafe's docs argue for: the model makes small factual
    judgments, the *policy* for turning them into a verdict lives in code where
    it can be read, tested and changed without touching a prompt.

    Each side is a disjunction -- one serious failing is enough to be at fault,
    so an average would let three mild answers bury one damning one.

    KNOWN BIAS: OP_FAULT has four questions and OTHER_FAULT has two, so `op`
    gets twice as many chances to cross the threshold. Under noise alone that
    tilts the fold toward YTA and ESH, independent of what the poster did. The
    confusion table in the report is the diagnostic: if the decomposed arm
    over-predicts YTA/ESH relative to the monolithic arm on the same posts,
    this is why, not the model. Kept as-is because dropping two OP questions to
    balance the count would cost real signal; measured rather than hidden.
    """
    op = max((probs[k] for k in OP_FAULT if k in probs), default=0.0)
    other = max((probs[k] for k in OTHER_FAULT if k in probs), default=0.0)
    if op >= at and other >= at:
        return "esh"
    if op >= at:
        return "yta"
    if other >= at:
        return "nta"
    return "nah"


def severity_target(verdict: str) -> float:
    """Where a verdict sits on SEVERITY_LEVELS, normalised to [0,1].

    Approximate by construction: Reddit labels the verdict, not a severity, so
    this maps the verdict onto the rubric rather than measuring against an
    independent human rating. Reported as such.
    """
    return {"nah": 0.0, "nta": 0.0, "esh": 2 / 3, "yta": 2 / 3}[verdict]
