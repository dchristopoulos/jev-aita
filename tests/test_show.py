"""The post viewer, with Jev stubbed out."""

import sys
from types import SimpleNamespace

from jevbench import show
from jevbench.client import ChoiceAns


def test_own_post_is_judged_live_without_a_forum_verdict(monkeypatch, capsys):
    def fake_ask(text, questions, model):
        assert text.startswith("AITA for X?\n\n")
        choices = ({"verdict": ChoiceAns("nta", {"nta": 0.9, "yta": 0.05, "esh": 0.03, "nah": 0.02}, 0.9)}
                   if "verdict" in questions else {})
        probs = {} if choices else {"poster_at_fault": 0.2, "other_at_fault": 0.8}
        return SimpleNamespace(probs=probs, choices=choices, latency_s=0.4,
                               billed_usd=2e-5, resolved_model="jev-test")

    monkeypatch.setattr(show, "ask", fake_ask)
    monkeypatch.setattr(sys, "argv", ["show", "--text", "AITA for X?\nI did Y."])
    assert show.main() == 0
    out = capsys.readouterr().out
    assert "Jev picked NTA" in out and "forum" not in out.lower()


def test_random_post_uses_benchmark_filters_and_reddits_verdict(monkeypatch):
    body = "x" * 500
    rows = [{"id": "leak", "title": "AITA?", "selftext_cleaned": body + " EDIT: I'm NTA",
             "link_flair_text": "Not the A-hole"},
            {"id": "info", "title": "AITA?", "selftext_cleaned": body,
             "link_flair_text": "Not enough info"},
            {"id": "ok", "title": "AITA for Y?", "selftext_cleaned": body,
             "link_flair_text": "Everyone Sucks", "score": 3}]
    monkeypatch.setattr(show, "get_json", lambda url: {"num_rows_total": 300,
                                                   "rows": [{"row": r} for r in rows]})
    item = show.fetch_random(__import__("random").Random(0))
    assert (item.id, item.verdict, item.title) == ("ok", "esh", "AITA for Y?")
