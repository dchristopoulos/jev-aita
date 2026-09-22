"""OpenRouter clients for Jev (Decisions API) and chat LLMs (Completions API).

Both return an Answer so the benchmark can treat them identically. The
asymmetry the benchmark exists to measure: Jev answers natively in constrained
probabilities, while an LLM has to be coaxed into emitting them and can refuse.
"""

from __future__ import annotations

import json
import math
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
# Any OpenAI-compatible server, LM Studio by default. Models are named
# `local/<id>` so they get their own rows and cost nothing.
LOCAL_PREFIX = "local/"


def local_chat_url() -> str:
    """LOCAL_LLM_URL as a full endpoint; a bare `http://host:1234` is fine too.

    A bare host goes to LM Studio's own /api/v0 endpoint rather than the
    OpenAI-compatible /v1 one: same request and answer, but it also reports
    tokens per second, time to first token, quantisation and runtime. Give a
    /v1 URL explicitly for any other OpenAI-compatible server.

    Read per call, not at import, so it can be pointed at another machine
    without restarting anything.
    """
    base = os.environ.get("LOCAL_LLM_URL", "http://localhost:1234").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith(("/v1", "/api/v0")):
        return base + "/chat/completions"
    return base + "/api/v0/chat/completions"

# $/M tokens. Jev bills output at zero; that asymmetry is the finding to test.
PRICING = {
    "~typesafe/jev-latest": (0.042, 0.0),
    "openai/gpt-5-nano#minimal": (0.025, 0.20),
    "openai/gpt-5-nano": (0.025, 0.20),
    "anthropic/claude-sonnet-5": (2.0, 10.0),
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
    def cost_usd(self) -> float:
        """What this request cost: the provider's own figure when it gives one."""
        if self.billed_usd:
            return self.billed_usd
        from .metrics import cost_usd

        in_m, out_m = PRICING.get(self.model, (0.0, 0.0))
        return cost_usd(self.prompt_tokens, self.completion_tokens, in_m, out_m)


def _post(url: str, body: dict, timeout: float = 30.0,
          retries: int = 3, local: bool = False) -> tuple[dict, float, int]:
    """POST with retry on 429/5xx. Returns response, total wait, attempts."""
    key = "" if local else api_key()
    if not local and "REPLACE-ME" in key:
        raise FatalApiError(
            "The .env file still has the placeholder key. Replace "
            "sk-or-v1-REPLACE-ME with your real key from openrouter.ai/keys."
        )
    if not local and not key:
        raise FatalApiError(
            "No API key. Either export OPENROUTER_API_KEY, or put it in a .env file "
            "at the repo root as OPENROUTER_API_KEY=sk-or-v1-... (.env is gitignored)."
        )

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {key}"} if key else {})},
    )
    last = None
    started = time.perf_counter()
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r), time.perf_counter() - started, attempt + 1
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
            if local:
                # Nothing listening: every remaining item would fail the same way.
                raise FatalApiError(
                    f"No local model server at {url}. Start LM Studio's server "
                    "(`lms server start`) and load the model first.") from e
            raise ApiError(f"Network error: {e}") from e
    raise ApiError(f"Exhausted {retries} retries: {last}")


def ask_jev(text: str, questions: dict[str, dict], model: str = "~typesafe/jev-latest") -> Answer:
    """One Decisions call. Every question sees the same state, evaluated in parallel."""
    data, dt, tries = _post(
        DECISIONS_URL, {"model": model, "state": text, "questions": questions})
    try:
        probs, choices = _split_answers(data["answers"], questions)
    except (KeyError, TypeError, ValueError) as e:
        raise ApiError(f"Unexpected Decisions response: {json.dumps(data)[:300]}") from e
    prompt_tok, completion_tok, billed = read_usage(data)
    return Answer(
        probs=probs,
        choices=choices,
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


def _split_answers(
    answers: dict, questions: dict[str, dict]
) -> tuple[dict[str, float], dict[str, ChoiceAns]]:
    """Pull each answer out by the type we asked for, not by guessing."""
    probs: dict[str, float] = {}
    choices: dict[str, ChoiceAns] = {}
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
        else:
            raise ValueError(f"unknown question type {q['type']!r}")
    return probs, choices


def production_verdict_prompt(question: dict) -> str:
    """A normal four-class classifier prompt for the chat baselines."""
    labels = "\n".join(f"{code.upper()}: {meaning}" for code, meaning in question["criteria"].items())
    return (
        "Predict the official r/AmItheAsshole post flair from the title and post text. "
        "Predict the community outcome, not your personal moral judgment. "
        "Use only the supplied text.\n\n"
        f"Labels:\n{labels}\n\n"
        "Return only a JSON object with a verdict field containing the probability "
        "of each label: {\"verdict\": {\"yta\": 0.0, \"nta\": 0.0, "
        "\"esh\": 0.0, \"nah\": 0.0}}. The four probabilities must sum to 1."
    )


def standard_choice_prompt(question: dict) -> str:
    labels = "\n".join(f"{n}. {code.upper()}: {meaning}"
                       for n, (code, meaning) in enumerate(question["criteria"].items(), 1))
    return (
        "You are an experienced r/AmItheAsshole reader. Read the title and post, "
        "then predict the official verdict the community would assign, not your "
        "personal moral judgment. Use only the supplied text.\n\n"
        f"Choose one:\n{labels}\n\nReply with only its number (1, 2, 3, or 4)."
    )


def _parse_standard_choice(content: str, question: dict) -> ChoiceAns:
    labels = list(question["criteria"])
    match = re.match(r"^\s*([1-4]|YTA|NTA|ESH|NAH)(?=\W|$)", content, re.I)
    token = match.group(1).lower() if match else ""
    choice = labels[int(token) - 1] if token in {"1", "2", "3", "4"} else token
    if choice not in labels:
        choice = UNPARSED_CHOICE
    return ChoiceAns(choice, {}, 0.0)


def split_variant(model: str) -> tuple[str, str]:
    """`openai/gpt-5-nano#minimal` -> ("openai/gpt-5-nano", "minimal").

    A reasoning model's thinking budget changes its latency and cost by more
    than an order of magnitude, so the budget is part of what is being
    measured. Keeping it in the label means both variants appear as separate
    rows instead of one silently overwriting the other.
    """
    base, _, effort = model.partition("#")
    return base, effort


def ask_llm(text: str, questions: dict[str, dict], model: str,
            label_only: bool = False) -> Answer:
    """The verdict question to a chat model, which must be asked to *emit* an answer.

    Parse failures are recorded rather than retried: an LLM that returns prose
    or refuses is a real cost of using a text model for a typed job, and
    retrying until it complies would flatter the baseline.
    """
    if set(questions) != {"verdict"} or questions["verdict"]["type"] != "choice":
        raise ValueError("chat models answer only the direct verdict question")
    base, effort = split_variant(model)
    verdict = questions["verdict"]
    system_prompt = (standard_choice_prompt(verdict) if label_only
                     else production_verdict_prompt(verdict))
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Title and post:\n{text}"}]
    body = {
        "model": base,
        "messages": messages,
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
    if base.startswith(LOCAL_PREFIX):
        # Local servers take the bare model id, may not know the reasoning
        # field, and a 27B model on a laptop needs far more than 30 s.
        body["model"] = base[len(LOCAL_PREFIX):]
        del body["reasoning"]
        if effort:
            # LM Studio honours the top-level OpenAI field and ignores both the
            # OpenRouter-style `reasoning` object and Qwen's /no_think. Measured
            # on qwen3.6-35b-a3b: "none" 0 reasoning tokens and ~1 s a post,
            # "low" ~1,400 tokens and ~17 s.
            body["reasoning_effort"] = effort
        data, dt, tries = _post(local_chat_url(), body, timeout=600.0, local=True)
    else:
        data, dt, tries = _post(CHAT_URL, body)
    prompt_tok, completion_tok, billed = read_usage(data)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ApiError(f"Unexpected chat response: {json.dumps(data)[:300]}") from e

    parse = _parse_standard_choice if label_only else _parse_distribution
    answer = parse(content or "", verdict)
    return Answer(
        probs={},
        choices={"verdict": answer},
        model=model,
        latency_s=dt,
        prompt_tokens=prompt_tok,
        completion_tokens=completion_tok,
        billed_usd=billed,
        n_questions=len(questions),
        raw=data,
        parse_failed=answer.choice == UNPARSED_CHOICE,
        attempts=tries,
        resolved_model=str(data.get("model", "")),
        request_id=str(data.get("id", "")),
    )


def _parse_distribution(content: str, question: dict) -> ChoiceAns:
    """Read `{"verdict": {"yta": p, ...}}` from a reply; anything else is unparsed.

    Four finite nonnegative numbers with a positive sum are accepted and
    rescaled to sum to 1, since chat models sometimes return relative weights
    such as 0.1, 0.85, 0.1, 0.05. A missing label, a non-number or an all-zero
    vector is a failure and counts against the model.
    """
    opts = list(question["criteria"])
    m = re.search(r"\{.*\}", content, re.S)   # tolerate code fences and stray prose
    try:
        v = json.loads(m.group(0)).get("verdict") if m else None
        dist = v.get("probabilities", v) if isinstance(v, dict) else None
        if isinstance(dist, dict) and set(dist) == set(opts):
            ps = {k: float(dist[k]) for k in opts}
            total = sum(ps.values())
            if all(math.isfinite(p) and p >= 0 for p in ps.values()) and total > 0:
                ps = {k: p / total for k, p in ps.items()}
                pick = max(opts, key=ps.get)
                return ChoiceAns(pick, ps, ps[pick])
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        pass
    return ChoiceAns(UNPARSED_CHOICE, {}, 0.0)


def ask(text: str, questions: dict[str, dict], model: str,
        label_only: bool = False) -> Answer:
    """Dispatch on model family so the runner treats every arm identically."""
    if model.startswith("~typesafe/"):
        if label_only:
            raise ValueError("label-only prompting is for chat models")
        return ask_jev(text, questions, model)
    return ask_llm(text, questions, model, label_only=label_only)
