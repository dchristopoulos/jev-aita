"""The runner, end to end, with the model calls faked out."""

import json
import sys
from dataclasses import asdict

import pytest

import jevbench.bench as bench
from jevbench.client import Answer, ChoiceAns, production_verdict_prompt
from jevbench.data import Item
from jevbench.questions import monolithic


def _fake_ask(text, questions, model, label_only=False):
    return Answer(probs={}, model=model, latency_s=0.1, prompt_tokens=10,
                  completion_tokens=2, n_questions=len(questions), billed_usd=1e-6,
                  choices={"verdict": ChoiceAns("nta", {"yta": .1, "nta": .7, "esh": .1,
                                                        "nah": .1}, .7)})


def _main(monkeypatch, tmp_path, *extra):
    sample = tmp_path / "sample.jsonl"
    sample.write_text("".join(
        json.dumps(asdict(Item(str(i), "t", "x", "nta", 0, []))) + "\n" for i in range(3)))
    out = tmp_path / "run.jsonl"
    monkeypatch.setattr(bench, "ask", _fake_ask)
    monkeypatch.setattr(sys, "argv", ["bench", "--data", str(sample), "--out", str(out),
                                      "--budget", "0", *extra])
    return bench.main(), out


def test_manifest_pins_timing_code_state_and_exact_prompts(monkeypatch, tmp_path):
    code, out = _main(monkeypatch, tmp_path, "--models", "~typesafe/jev-latest",
                      "openai/gpt-5-nano#minimal")
    assert code == 0
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["started"] <= meta["finished"]
    assert isinstance(meta["git_dirty"], bool)
    assert meta["prompts"] == {
        "jev/monolithic": monolithic(),
        "chat/monolithic": production_verdict_prompt(monolithic()["verdict"])}
    assert meta["ok"] == 6 and meta["errors"] == 0
    assert len(out.read_text().splitlines()) == 6


def test_runner_refuses_to_overwrite_a_log(monkeypatch, tmp_path):
    (tmp_path / "run.jsonl").write_text("")
    code, _ = _main(monkeypatch, tmp_path, "--models", "~typesafe/jev-latest")
    assert code == 1


@pytest.mark.parametrize("models,arm", [
    (["~typesafe/jev-latest"], "standard_choice"),
    (["openai/gpt-5-nano"], "balanced"),
])
def test_arms_are_limited_to_the_models_they_were_designed_for(monkeypatch, tmp_path, models, arm):
    with pytest.raises(SystemExit):
        _main(monkeypatch, tmp_path, "--models", *models, "--arms", arm)
