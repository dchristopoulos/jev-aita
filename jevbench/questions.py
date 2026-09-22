"""The two question sets the final benchmark sends to Jev.

  monolithic  one Choice: "which verdict would the subreddit reach?"
  balanced    two yes/no questions about blame, combined into four verdicts

The question text here is exactly what the final runs sent; changing it makes
new runs incomparable with the logs in runs/.
"""

from __future__ import annotations

from .data import VERDICTS

VERDICT_QUESTION = (
    "You are judging a post from r/AmItheAsshole. "
    "Which verdict would the subreddit reach about the poster?"
)


def _noul(instructions: str, yes: str, no: str) -> dict:
    return {"type": "noul", "instructions": instructions,
            "criteria": {"true": yes, "false": no}}


def monolithic() -> dict[str, dict]:
    """Ask for the verdict directly."""
    return {"verdict": {"type": "choice", "instructions": VERDICT_QUESTION,
                        "criteria": dict(VERDICTS)}}


def balanced() -> dict[str, dict]:
    """Two symmetric fault questions for a transparent four-way rule."""
    return {
        "poster_at_fault": _noul(
            "Would r/AmItheAsshole readers judge the poster blameworthy in this conflict?",
            "The poster's own conduct warrants blame", "The poster's conduct does not warrant blame"),
        "other_at_fault": _noul(
            "Would r/AmItheAsshole readers judge at least one other person blameworthy in this conflict?",
            "At least one other person's conduct warrants blame",
            "No other person's conduct warrants blame"),
    }


def balanced_distribution(probs: dict[str, float]) -> dict[str, float]:
    """Four verdict probabilities under a simple independence assumption."""
    op, other = probs["poster_at_fault"], probs["other_at_fault"]
    if not (0 <= op <= 1 and 0 <= other <= 1):
        raise ValueError("fault probabilities must be in [0,1]")
    return {"yta": op * (1 - other), "nta": (1 - op) * other,
            "esh": op * other, "nah": (1 - op) * (1 - other)}
