"""OpenRouter clients for Jev (Decisions API) and chat LLMs (Completions API).

Both return an Answer so the benchmark can treat them identically. The
asymmetry the benchmark exists to measure: Jev answers natively in constrained
probabilities, while an LLM has to be coaxed into emitting them and can refuse.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


# What a Choice answer becomes when the model returns something that is not one
# of the supplied options. Deliberately not a real option: defaulting to the
# first one would let a failure score correct whenever that option happened to
# be right, quietly discounting the cost of failing.
UNPARSED_CHOICE = "__unparsed__"

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# $/M tokens. Jev bills output at zero; that asymmetry is the finding to test.
PRICING = {
    "~typesafe/jev-latest": (0.042, 0.0),
    "openai/gpt-5-nano#minimal": (0.025, 0.20),
    "openai/gpt-5-nano": (0.025, 0.20),
    "anthropic/claude-sonnet-5": (3.0, 15.0),
}


def api_key() -> str:
    """The key, from the environment or a gitignored .env at the repo root.

    The .env path exists so a key never has to be pasted into a terminal
    transcript or a chat log to run this.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip("\"'")
    return ""


class ApiError(RuntimeError):
    """A request failed. The runner records it and carries on."""


class FatalApiError(ApiError):
    """Nothing will work until a human fixes it: no key, bad key, no credit.

    Separate from ApiError so the runner can stop immediately instead of
    printing the same message once per post for the rest of the run.
    """


@dataclass
class ChoiceAns:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass
class ScoreAns:
    position: float                   # normalised to [0,1] across the levels
    raw_score: float
    probabilities: dict[str, float]
    confidence: float


def read_usage(data: dict) -> tuple[int, int, float]:
    """(prompt, completion, billed_usd) from a response, whichever names it uses.

    The Decisions API reports `input_tokens`/`output_tokens`; chat completions
    report `prompt_tokens`/`completion_tokens`. Reading only one pair silently
    logs zero tokens and zero cost for the other -- which is exactly what this
    benchmark is trying to measure.

    `cost` is what the provider actually billed. Preferred over multiplying by
    a local price table, which goes stale the moment a price changes.
    """
    u = data.get("usage") or {}
    prompt = int(u.get("prompt_tokens", u.get("input_tokens", 0)) or 0)
    completion = int(u.get("completion_tokens", u.get("output_tokens", 0)) or 0)
    billed = float(u.get("cost", 0.0) or 0.0)
    return prompt, completion, billed


@dataclass
class Answer:
    """One model's response to one request, which may hold several questions."""

    probs: dict[str, float]      # noul id -> probability in [0,1]
    choices: dict[str, ChoiceAns]
    scores: dict[str, ScoreAns]
    model: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    n_questions: int
    raw: dict = field(repr=False, default_factory=dict)
    parse_failed: bool = False   # model emitted something unusable
    attempts: int = 1            # >1 means it was retried; latency is the successful try
    resolved_model: str = ""     # what the API says it actually served, e.g. jev-1.13.0
    request_id: str = ""         # provider request id, for auditing a specific answer
    billed_usd: float = 0.0      # what the provider says it charged, 0 if not reported

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def all_confidence(self) -> dict[str, float]:
        """Real confidence, available only for Choice and Score."""
        return {**{k: v.confidence for k, v in self.choices.items()},
                **{k: v.confidence for k, v in self.scores.items()}}

    @property
    def cost_usd(self) -> float:
        """What this request cost: the provider's own figure when it gives one."""
        if self.billed_usd:
            return self.billed_usd
        from .metrics import cost_usd

        in_m, out_m = PRICING.get(self.model, (0.0, 0.0))
        return cost_usd(self.prompt_tokens, self.completion_tokens, in_m, out_m)


def _post(url: str, body: dict, timeout: float = 30.0,
          retries: int = 3) -> tuple[dict, float, int]:
    """POST with retry on 429/5xx. Returns (json, wall_seconds, attempts).

    Latency is measured around the HTTP call only, so it is comparable across
    models. A retried call reports the successful attempt's latency; including
    backoff sleep would misrepresent steady-state latency.
    """
    key = api_key()
    if "REPLACE-ME" in key:
        raise FatalApiError(
            "The .env file still has the placeholder key. Replace "
            "sk-or-v1-REPLACE-ME with your real key from openrouter.ai/keys."
        )
    if not key:
        raise FatalApiError(
            "No API key. Either export OPENROUTER_API_KEY, or put it in a .env file "
            "at the repo root as OPENROUTER_API_KEY=sk-or-v1-... (.env is gitignored)."
        )

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    last = None
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r), time.perf_counter() - t0, attempt + 1
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                last = e
                time.sleep(2**attempt)
                continue
            if e.code == 402:
                raise FatalApiError(f"Out of credit on OpenRouter: {detail}") from e
            if e.code in (401, 403):
                raise FatalApiError(f"Key rejected (HTTP {e.code}): {detail}") from e
            raise ApiError(f"HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                last = e
                time.sleep(2**attempt)
                continue
            raise ApiError(f"Network error: {e}") from e
    raise ApiError(f"Exhausted {retries} retries: {last}")


def ask_jev(text: str, questions: dict[str, dict], model: str = "~typesafe/jev-latest") -> Answer:
    """One Decisions call. Every question sees the same state, evaluated in parallel."""
    data, dt, tries = _post(
        DECISIONS_URL, {"model": model, "state": text, "questions": questions})
    try:
        probs, choices, scores = _split_answers(data["answers"], questions)
    except (KeyError, TypeError, ValueError) as e:
        raise ApiError(f"Unexpected Decisions response: {json.dumps(data)[:300]}") from e
    prompt_tok, completion_tok, billed = read_usage(data)
    return Answer(
        probs=probs,
        choices=choices,
        scores=scores,
        model=model,
        latency_s=dt,
        prompt_tokens=prompt_tok,
        completion_tokens=completion_tok,
        billed_usd=billed,
        n_questions=len(questions),
        raw=data,
        attempts=tries,
        resolved_model=str(data.get("model", "")),
        request_id=str(data.get("id", "")),
    )


def normalise_score(raw: float, legend: dict | list | None, n_levels: int) -> float:
    """Map a Score onto [0,1] across its levels.

    `legend` repeats the levels by number, but the docs do not pin whether
    numbering starts at 0 or 1, and a score may fall *between* two levels. Use
    the legend when it carries numbers; otherwise assume 1..n.

    That assumption is not guessed at per call on purpose. Indexing is a
    property of the API, not of one answer, so inferring it from whether a
    single score happened to fall below 1.0 would scale some answers in a run
    differently from others. `raw_score` is kept in the run log and
    `jevbench.report` re-derives the convention once, across every score it
    saw. This value is the best guess for live use, such as the demo.
    """
    lo, hi = 1.0, float(n_levels)
    nums = _legend_numbers(legend)
    if nums:
        lo, hi = min(nums), max(nums)
    if hi <= lo:
        return 0.0
    return min(max((float(raw) - lo) / (hi - lo), 0.0), 1.0)


def _legend_numbers(legend: dict | list | None) -> list[float]:
    """Level numbers from a legend, whichever side of it they are on.

    The docs say `legend` "repeats the levels by number" but do not pin the
    shape, so accept {number: label} and {label: number}, and give up rather
    than guess when neither side is numeric.
    """
    if not isinstance(legend, dict) or not legend:
        return []
    for side in (legend.keys(), legend.values()):
        try:
            return [float(x) for x in side]
        except (TypeError, ValueError):
            continue
    return []


def _split_answers(
    answers: dict, questions: dict[str, dict]
) -> tuple[dict[str, float], dict[str, ChoiceAns], dict[str, ScoreAns]]:
    """Pull each answer out by the type we asked for, not by guessing."""
    probs: dict[str, float] = {}
    choices: dict[str, ChoiceAns] = {}
    scores: dict[str, ScoreAns] = {}
    for qid, q in questions.items():
        a = answers[qid]
        if q["type"] == "noul":
            probs[qid] = float(a["noul"])
        elif q["type"] == "choice":
            choices[qid] = ChoiceAns(
                choice=str(a["choice"]),
                probabilities={str(k): float(v) for k, v in a.get("probabilities", {}).items()},
                confidence=float(a["confidence"]),
            )
        elif q["type"] == "score":
            scores[qid] = ScoreAns(
                position=normalise_score(a["score"], a.get("legend"), len(q["criteria"])),
                raw_score=float(a["score"]),
                probabilities={str(k): float(v) for k, v in a.get("probabilities", {}).items()},
                confidence=float(a["confidence"]),
            )
        else:
            raise ValueError(f"unknown question type {q['type']!r}")
    return probs, choices, scores


def _llm_prompt(questions: dict[str, dict]) -> str:
    """Ask a chat model for the same judgments, as strict JSON.

    The fairest available framing: one call, same state, all questions, so the
    LLM gets the same batching advantage Jev gets. Confidence has to be
    self-reported, which is precisely the asymmetry under test -- Jev derives
    it from a real distribution, an LLM is guessing at its own certainty.
    """
    lines = []
    for qid, q in questions.items():
        if q["type"] == "noul":
            lines.append(f'  "{qid}": <probability 0.00-1.00>,   // {q["instructions"]}')
        elif q["type"] == "choice":
            opts = "|".join(q["criteria"])
            lines.append(
                f'  "{qid}": {{"choice": "<{opts}>", "confidence": <0.00-1.00>}},   '
                f'// {q["instructions"]}'
            )
        else:
            n = len(q["criteria"])
            levels = "; ".join(f"{i+1}={lvl}" for i, lvl in enumerate(q["criteria"]))
            lines.append(
                f'  "{qid}": {{"level": <1-{n}, may be fractional>, "confidence": <0.00-1.00>}},   '
                f'// {q["instructions"]} Levels: {levels}'
            )
    return (
        "For the comment below, answer every question about how a majority of human "
        "readers would judge it.\n"
        "Reply with ONLY a JSON object, no prose, no code fence:\n{\n"
        + "\n".join(lines).rstrip(",")
        + "\n}\n\nComment:\n{text}"
    )


def split_variant(model: str) -> tuple[str, str]:
    """`openai/gpt-5-nano#minimal` -> ("openai/gpt-5-nano", "minimal").

    A reasoning model's thinking budget changes its latency and cost by more
    than an order of magnitude, so the budget is part of what is being
    measured. Keeping it in the label means both variants appear as separate
    rows instead of one silently overwriting the other.
    """
    base, _, effort = model.partition("#")
    return base, effort


def ask_llm(text: str, questions: dict[str, dict], model: str) -> Answer:
    """Same questions to a chat model, which must be asked to *emit* the answers.

    Parse failures are recorded rather than retried: an LLM that returns prose
    or refuses is a real cost of using a text model for a typed job, and
    retrying until it complies would flatter the baseline.
    """
    base, effort = split_variant(model)
    prompt = _llm_prompt(questions).replace("{text}", text)
    body = {
        "model": base,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        # Generous on purpose. Reasoning models spend this budget *before*
        # emitting anything, so a tight cap produces finish_reason="length"
        # with content=None on every call -- which scores as the model
        # refusing to answer when in fact it was never given room to.
        "max_tokens": 2000 + 150 * len(questions),
        # Keep reasoning short rather than banning it: the comparison is about
        # answering the same questions, and an unbounded chain of thought makes
        # the latency and cost columns measure thinking budget, not the task.
        "reasoning": {"effort": effort or "low"},
    }
    data, dt, tries = _post(CHAT_URL, body)
    prompt_tok, completion_tok, billed = read_usage(data)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ApiError(f"Unexpected chat response: {json.dumps(data)[:300]}") from e

    probs, choices, scores, failed = _parse_llm(content or "", questions)
    return Answer(
        probs=probs,
        choices=choices,
        scores=scores,
        model=model,
        latency_s=dt,
        prompt_tokens=prompt_tok,
        completion_tokens=completion_tok,
        billed_usd=billed,
        n_questions=len(questions),
        raw=data,
        parse_failed=failed,
        attempts=tries,
        resolved_model=str(data.get("model", "")),
        request_id=str(data.get("id", "")),
    )


def _parse_llm(
    content: str, questions: dict[str, dict]
) -> tuple[dict[str, float], dict[str, ChoiceAns], dict[str, ScoreAns], bool]:
    """Recover typed answers from an LLM reply.

    Anything missing or unusable falls back to a neutral default and sets the
    failure flag, so the report can count how often this happened rather than
    silently scoring a refusal as a confident answer. Jev cannot fail this way;
    that asymmetry is the structural argument for a typed model.
    """
    obj = {}
    m = re.search(r"\{.*\}", content, re.S)   # tolerate code fences and stray prose
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = {}

    probs: dict[str, float] = {}
    choices: dict[str, ChoiceAns] = {}
    scores: dict[str, ScoreAns] = {}
    failed = False

    for qid, q in questions.items():
        v = obj.get(qid)
        if q["type"] == "noul":
            try:
                f = float(v)
            except (TypeError, ValueError):
                probs[qid], failed = 0.5, True
                continue
            if not 0.0 <= f <= 1.0:
                failed = True
            probs[qid] = min(max(f, 0.0), 1.0)

        elif q["type"] == "choice":
            opts = list(q["criteria"])
            pick = v.get("choice") if isinstance(v, dict) else v
            conf = _conf(v)
            if pick not in opts:
                # Hallucinated an option outside the list -- something Jev's
                # constrained output makes impossible.
                pick, conf, failed = UNPARSED_CHOICE, 0.0, True
            choices[qid] = ChoiceAns(choice=str(pick), probabilities={}, confidence=conf)

        else:
            n = len(q["criteria"])
            lvl = v.get("level") if isinstance(v, dict) else v
            try:
                raw = float(lvl)
            except (TypeError, ValueError):
                raw, failed = (n + 1) / 2, True
            scores[qid] = ScoreAns(
                position=normalise_score(raw, None, n),
                raw_score=raw,
                probabilities={},
                confidence=_conf(v),
            )

    return probs, choices, scores, failed


def _conf(v: object) -> float:
    """Self-reported confidence, defaulting to 0.5 when absent or unusable."""
    if not isinstance(v, dict):
        return 0.5
    try:
        return min(max(float(v.get("confidence", 0.5)), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.5


def ask(text: str, questions: dict[str, dict], model: str) -> Answer:
    """Dispatch on model family so the runner treats every arm identically."""
    if model.startswith("~typesafe/"):
        return ask_jev(text, questions, model)
    return ask_llm(text, questions, model)   # label keeps any #effort variant
