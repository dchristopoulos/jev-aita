import json
import pytest

from jevbench.client import _parse_llm
from jevbench.questions import full

NOUL = {"toxicity": {"type": "noul", "instructions": "?", "criteria": {}}}
TWO_NOULS = {**NOUL, "threat": {"type": "noul", "instructions": "?", "criteria": {}}}


def _nouls(content, questions=NOUL):
    probs, _, _, failed = _parse_llm(content, questions)
    return probs, failed


def test_parses_json_inside_a_code_fence_and_prose():
    probs, failed = _nouls('Sure!\n```json\n{"toxicity": 0.9}\n```')
    assert probs["toxicity"] == 0.9 and not failed


def test_missing_key_falls_back_to_neutral_and_is_flagged():
    probs, failed = _nouls('{"toxicity": 0.9}', TWO_NOULS)
    assert probs["threat"] == 0.5 and failed


def test_out_of_range_is_clamped_and_flagged():
    probs, failed = _nouls('{"toxicity": 1.4}')
    assert probs["toxicity"] == 1.0 and failed


def test_refusal_prose_is_flagged_not_crashed():
    probs, failed = _nouls("I can't assess that.")
    assert probs["toxicity"] == 0.5 and failed


def test_non_numeric_value_is_flagged():
    probs, failed = _nouls('{"toxicity": "no"}')
    assert probs["toxicity"] == 0.5 and failed


def test_hallucinated_choice_is_flagged_and_zero_confidence():
    from jevbench.client import _parse_llm
    from jevbench.questions import full

    q = {"verdict": full()["verdict"]}
    _, choices, _, failed = _parse_llm('{"verdict": {"choice": "maybe"}}', q)
    assert failed and choices["verdict"].confidence == 0.0


def test_score_normalisation_handles_both_index_conventions():
    from jevbench.client import normalise_score

    assert normalise_score(1.0, None, 4) == 0.0
    assert normalise_score(4.0, None, 4) == 1.0
    assert normalise_score(1.5, {"0": "a", "1": "b", "2": "c", "3": "d"}, 4) == 0.5


def test_score_normalisation_survives_a_degenerate_legend():
    from jevbench.client import normalise_score

    assert normalise_score(2.0, {"1": "only"}, 1) == 0.0


def test_unparsed_choice_can_never_score_correct():
    """A parse failure must not accidentally be right.

    Defaulting to the first option would let a failure score correct whenever
    that option happened to be the true verdict -- 35% of a stratified sample.
    """
    from jevbench.client import UNPARSED_CHOICE
    from jevbench.data import VERDICTS

    assert UNPARSED_CHOICE not in VERDICTS


def test_hallucinated_choice_becomes_the_sentinel():
    from jevbench.client import UNPARSED_CHOICE, _parse_llm

    q = {"verdict": full()["verdict"]}
    _, choices, _, failed = _parse_llm('{"verdict": {"choice": "probably yta"}}', q)
    assert failed and choices["verdict"].choice == UNPARSED_CHOICE


def test_missing_key_is_fatal_not_per_item(monkeypatch):
    """Without this, a keyless run prints the same error once per post.

    api_key() is patched rather than just clearing the env var: a real .env at
    the repo root would otherwise satisfy the lookup and this would pass for
    the wrong reason on the maintainer's machine and fail in CI.
    """
    import jevbench.client as client

    monkeypatch.setattr(client, "api_key", lambda: "")
    with pytest.raises(client.FatalApiError):
        client._post("https://example.invalid", {})


def test_legend_numbers_read_either_side():
    from jevbench.client import _legend_numbers

    assert _legend_numbers({"1": "Civil", "2": "Rude"}) == [1.0, 2.0]
    assert _legend_numbers({"Civil": 1, "Rude": 2}) == [1.0, 2.0]
    assert _legend_numbers({"a": "x"}) == []
    assert _legend_numbers(None) == []


def test_score_normalisation_does_not_guess_indexing_per_call():
    """Indexing is a property of the API; the report decides it once per run."""
    from jevbench.client import normalise_score

    # A sub-1.0 score must not silently flip this call to 0-indexed scaling.
    assert normalise_score(3.0, None, 4) == normalise_score(3.0, None, 4)
    assert normalise_score(1.0, None, 4) == 0.0


def test_placeholder_key_fails_with_a_useful_message(monkeypatch):
    """A forgotten placeholder should say so, not surface as a bare 401."""
    from jevbench.client import FatalApiError, _post

    import jevbench.client as client

    monkeypatch.setattr(client, "api_key", lambda: "sk-or-v1-REPLACE-ME")
    with pytest.raises(FatalApiError, match="placeholder"):
        _post("https://example.invalid", {})


def test_usage_is_read_under_both_naming_conventions():
    """Regression: the Decisions API and chat completions name these differently.

    Reading only one pair logs zero tokens and zero cost for the other, which
    silently empties the cost comparison this benchmark exists to make.
    """
    from jevbench.client import read_usage

    decisions = {"usage": {"input_tokens": 1336, "output_tokens": 217, "cost": 5.6112e-05}}
    chat = {"usage": {"prompt_tokens": 900, "completion_tokens": 40, "cost": 0.0001}}
    assert read_usage(decisions) == (1336, 217, 5.6112e-05)
    assert read_usage(chat) == (900, 40, 0.0001)
    assert read_usage({}) == (0, 0, 0.0)


def test_billed_cost_is_preferred_over_the_local_price_table():
    """Prices change; the provider's own figure does not go stale."""
    from jevbench.client import Answer

    a = Answer(probs={}, choices={}, scores={}, model="~typesafe/jev-latest",
               latency_s=0.1, prompt_tokens=1336, completion_tokens=217,
               n_questions=1, billed_usd=5.6112e-05)
    assert a.cost_usd == 5.6112e-05


def test_falls_back_to_the_price_table_when_cost_is_absent():
    from jevbench.client import Answer

    a = Answer(probs={}, choices={}, scores={}, model="~typesafe/jev-latest",
               latency_s=0.1, prompt_tokens=1_000_000, completion_tokens=999,
               n_questions=1)
    assert a.cost_usd == pytest.approx(0.042)


def test_reasoning_variant_is_parsed_but_kept_in_the_label():
    """Thinking budget changes latency and cost by >10x, so it is part of the
    identity of what was measured, not a hidden setting."""
    from jevbench.client import split_variant

    assert split_variant("openai/gpt-5-nano#minimal") == ("openai/gpt-5-nano", "minimal")
    assert split_variant("openai/gpt-5-nano") == ("openai/gpt-5-nano", "")


def test_local_model_needs_no_key_and_hits_the_local_server(monkeypatch):
    import jevbench.client as client
    from jevbench.questions import monolithic

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
                "choices": [{"message": {"content": '{"verdict": {"choice": "nta", "confidence": 0.7}}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
        return R()

    monkeypatch.setattr(client.urllib.request, "urlopen", fake_urlopen)
    a = client.ask("a post", monolithic(), "local/qwen-27b")
    assert seen["url"] == client.LOCAL_URL and seen["auth"] is None
    assert seen["body"]["model"] == "qwen-27b" and "reasoning" not in seen["body"]
    assert seen["timeout"] >= 300
    assert a.model == "local/qwen-27b" and a.cost_usd == 0.0
    assert a.choices["verdict"].choice == "nta"
