import hashlib
import json

import pytest

import jevbench.client as client
from jevbench.client import (UNPARSED_CHOICE, Answer, _parse_distribution, _parse_standard_choice,
                             production_verdict_prompt, read_usage, split_variant,
                             standard_choice_prompt)
from jevbench.data import VERDICTS
from jevbench.questions import balanced, monolithic

VERDICT = monolithic()["verdict"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_prompts_are_pinned():
    """The final logs in runs/ come from this text; editing it breaks comparability.

    The chat prompts match the logged prompt token counts
    (scripts/check_prompt_tokens.py). Change a prompt only for a new,
    separately named run.
    """
    assert _sha(production_verdict_prompt(VERDICT)) == \
        "e90ebd4e1a5ca27c0d2190175bd0962c37aa90379986f5b2e46735dea9f1e447"
    assert _sha(standard_choice_prompt(VERDICT)) == \
        "a7a5fc05228326237b17ee839086655c95b0315d5d4ec5db888761da0cdaffce"
    assert _sha(json.dumps(monolithic(), sort_keys=True)) == \
        "76fcc97978069f5fe737b667225b9f4a7ac91fe69cad7843d3bdbde6cc63dace"
    assert _sha(json.dumps(balanced(), sort_keys=True)) == \
        "c44708a3c577b75b8a8290602f75433c769851dea64eb80e1e26c9baaa6f1c32"


def test_chat_prompts_carry_every_label_definition_jev_sees():
    # Fairness guard: if Jev is told what "nah" means and an LLM only sees the
    # code, the comparison measures the prompt rather than the model.
    for prompt in (production_verdict_prompt(VERDICT), standard_choice_prompt(VERDICT)):
        assert all(meaning in prompt for meaning in VERDICTS.values())


@pytest.mark.parametrize("content", [
    '{"verdict": {"yta": 0.1, "nta": 0.7, "esh": 0.1, "nah": 0.1}}',
    'Sure!\n```json\n{"verdict": {"yta": 0.1, "nta": 0.7, "esh": 0.1, "nah": 0.1}}\n```',
    '{"verdict": {"probabilities": {"yta": 0.1, "nta": 0.7, "esh": 0.1, "nah": 0.1}}}',
])
def test_distribution_is_read_through_fences_and_prose(content):
    a = _parse_distribution(content, VERDICT)
    assert a.choice == "nta" and a.probabilities["nta"] == pytest.approx(0.7)


def test_relative_weights_are_rescaled_to_sum_to_one():
    a = _parse_distribution('{"verdict": {"yta": 0.1, "nta": 0.85, "esh": 0.1, "nah": 0.05}}',
                            VERDICT)
    assert a.choice == "nta" and sum(a.probabilities.values()) == pytest.approx(1)


@pytest.mark.parametrize("content", [
    "I can't assess that.",
    '{"verdict": {"yta": 0, "nta": 0, "esh": 0, "nah": 0}}',
    '{"verdict": {"yta": 0.5, "nta": 0.5, "esh": 0}}',
    '{"verdict": {"yta": -0.1, "nta": 0.9, "esh": 0.1, "nah": 0.1}}',
    '{"verdict": {"yta": "high", "nta": 0.9, "esh": 0.1, "nah": 0.1}}',
    '{"verdict": {"choice": "nta", "confidence": 0.9}}',
    '[1, 2, 3]',
    '{"verdict": {bad json',
])
def test_unusable_replies_are_unparsed_not_guessed(content):
    a = _parse_distribution(content, VERDICT)
    assert a.choice == UNPARSED_CHOICE and not a.probabilities


def test_unparsed_choice_can_never_score_correct():
    """Defaulting to the first option would let a failure score correct whenever
    that option happened to be the true verdict."""
    assert UNPARSED_CHOICE not in VERDICTS


def test_label_only_parser_accepts_numbers_and_codes():
    assert _parse_standard_choice("2", VERDICT).choice == "nta"
    assert _parse_standard_choice("NTA", VERDICT).choice == "nta"
    assert _parse_standard_choice("not sure", VERDICT).choice == UNPARSED_CHOICE


def _fake_chat(sent: dict, content: str):
    def fake_post(url, body, timeout=30.0, retries=3, local=False):
        sent.update(body, url=url)
        return ({"choices": [{"message": {"content": content}}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 30}}, 0.1, 1)
    return fake_post


def test_chat_request_is_a_system_prompt_plus_the_post(monkeypatch):
    sent = {}
    monkeypatch.setattr(client, "_post", _fake_chat(
        sent, '{"verdict":{"yta":0.1,"nta":0.7,"esh":0.1,"nah":0.1}}'))
    a = client.ask("Title\n\nBody", monolithic(), "openai/gpt-5-nano#minimal")
    assert sent["messages"] == [
        {"role": "system", "content": production_verdict_prompt(VERDICT)},
        {"role": "user", "content": "Title and post:\nTitle\n\nBody"}]
    assert sent["model"] == "openai/gpt-5-nano" and sent["temperature"] == 0
    assert sent["reasoning"] == {"effort": "minimal"}
    assert a.choices["verdict"].choice == "nta" and not a.parse_failed


def test_label_only_request_uses_the_numbered_prompt(monkeypatch):
    sent = {}
    monkeypatch.setattr(client, "_post", _fake_chat(sent, "2"))
    a = client.ask("a post", monolithic(), "openai/gpt-5-nano#minimal", label_only=True)
    assert sent["messages"][0]["content"] == standard_choice_prompt(VERDICT)
    assert a.choices["verdict"].choice == "nta" and not a.choices["verdict"].probabilities


def test_chat_models_only_answer_the_verdict_question():
    with pytest.raises(ValueError):
        client.ask("a post", balanced(), "openai/gpt-5-nano")
    with pytest.raises(ValueError):
        client.ask("a post", monolithic(), "~typesafe/jev-latest", label_only=True)


def test_jev_answers_are_read_by_question_type(monkeypatch):
    sent = {}

    def fake_post(url, body, timeout=30.0, retries=3, local=False):
        sent.update(body, url=url)
        return ({"model": "typesafe/jev-1.13", "id": "req-1",
                 "answers": {"poster_at_fault": {"noul": 0.8}, "other_at_fault": {"noul": 0.3}},
                 "usage": {"input_tokens": 900, "output_tokens": 20, "cost": 4e-5}}, 0.2, 1)

    monkeypatch.setattr(client, "_post", fake_post)
    a = client.ask("a post", balanced(), "~typesafe/jev-latest")
    assert sent["url"] == client.DECISIONS_URL and sent["questions"] == balanced()
    assert a.probs == {"poster_at_fault": 0.8, "other_at_fault": 0.3}
    assert a.resolved_model == "typesafe/jev-1.13" and a.cost_usd == 4e-5


def test_missing_key_is_fatal_not_per_item(monkeypatch):
    """Without this, a keyless run prints the same error once per post.

    api_key() is patched rather than just clearing the env var: a real .env at
    the repo root would otherwise satisfy the lookup and this would pass for
    the wrong reason on the maintainer's machine and fail in CI.
    """
    monkeypatch.setattr(client, "api_key", lambda: "")
    with pytest.raises(client.FatalApiError):
        client._post("https://example.invalid", {})


def test_placeholder_key_fails_with_a_useful_message(monkeypatch):
    monkeypatch.setattr(client, "api_key", lambda: "sk-or-v1-REPLACE-ME")
    with pytest.raises(client.FatalApiError, match="placeholder"):
        client._post("https://example.invalid", {})


def test_usage_is_read_under_both_naming_conventions():
    """Regression: the Decisions API and chat completions name these differently."""
    decisions = {"usage": {"input_tokens": 1336, "output_tokens": 217, "cost": 5.6112e-05}}
    chat = {"usage": {"prompt_tokens": 900, "completion_tokens": 40, "cost": 0.0001}}
    assert read_usage(decisions) == (1336, 217, 5.6112e-05)
    assert read_usage(chat) == (900, 40, 0.0001)
    assert read_usage({}) == (0, 0, 0.0)


def _answer(**kw) -> Answer:
    return Answer(probs={}, choices={}, model="~typesafe/jev-latest", latency_s=0.1,
                  n_questions=1, **kw)


def test_billed_cost_is_preferred_over_the_local_price_table():
    assert _answer(prompt_tokens=1336, completion_tokens=217,
                   billed_usd=5.6112e-05).cost_usd == 5.6112e-05


def test_falls_back_to_the_price_table_when_cost_is_absent():
    assert _answer(prompt_tokens=1_000_000, completion_tokens=999).cost_usd == \
        pytest.approx(0.042)


def test_reasoning_variant_is_parsed_but_kept_in_the_label():
    assert split_variant("openai/gpt-5-nano#minimal") == ("openai/gpt-5-nano", "minimal")
    assert split_variant("openai/gpt-5-nano") == ("openai/gpt-5-nano", "")


def test_local_model_needs_no_key_and_hits_the_local_server(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
    monkeypatch.setattr(client, "api_key", lambda: (_ for _ in ()).throw(
        AssertionError("a local model must not need an OpenRouter key")))
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"], seen["auth"] = req.full_url, req.get_header("Authorization")
        seen["body"], seen["timeout"] = json.loads(req.data), timeout

        class R:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps({
                "choices": [{"message": {"content": '{"verdict": {"yta": 0.1, "nta": 0.7, "esh": 0.1, "nah": 0.1}}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
        return R()

    monkeypatch.setattr(client.urllib.request, "urlopen", fake_urlopen)
    a = client.ask("a post", monolithic(), "local/qwen-27b#none")
    assert seen["url"] == "http://localhost:1234/api/v0/chat/completions" and seen["auth"] is None
    assert seen["body"]["model"] == "qwen-27b" and "reasoning" not in seen["body"]
    assert seen["body"]["reasoning_effort"] == "none"
    assert seen["timeout"] >= 300
    assert a.model == "local/qwen-27b#none" and a.cost_usd == 0.0
    assert a.choices["verdict"].choice == "nta"


def test_local_url_accepts_a_bare_host(monkeypatch):
    lms = "http://10.0.0.5:1234/api/v0/chat/completions"
    oai = "http://10.0.0.5:1234/v1/chat/completions"
    for given, want in [("http://10.0.0.5:1234", lms), ("http://10.0.0.5:1234/", lms),
                        ("http://10.0.0.5:1234/api/v0", lms), ("http://10.0.0.5:1234/v1", oai),
                        (oai, oai)]:
        monkeypatch.setenv("LOCAL_LLM_URL", given)
        assert client.local_chat_url() == want, given
