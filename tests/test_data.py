"""The model must reach its own verdict, not read one off the page."""

from dataclasses import asdict

from jevbench.data import Item, leaks_verdict


def test_verdict_edits_are_detected():
    assert leaks_verdict("EDIT: ok everyone, seems like I'm NTA here")
    assert leaks_verdict("so am i yta for this?")
    assert leaks_verdict("verdict was ESH apparently")
    assert leaks_verdict("UPDATE: Reddit says not the asshole")
    assert leaks_verdict("Apparently everyone sucks here")
    assert leaks_verdict("UPDATE: you were right, I am a raging asshole")


def test_ordinary_posts_are_not_false_positives():
    assert not leaks_verdict("aita for buying my cousins a disney+ subscription")
    assert not leaks_verdict("my sister in law was upset about the natal chart")
    assert not leaks_verdict("i said no thanks and left")


def test_only_title_and_text_are_ever_sent_to_the_model():
    """The verdict, the score and the comments exist for the reader, not the model."""
    item = Item(id="1", title="aita for x", text="the story", verdict="yta",
                score=500, top_comments=["yta obviously", "hard yta"])
    assert item.body == "aita for x\n\nthe story"
    for secret in (item.verdict, str(item.score), *item.top_comments):
        assert secret not in item.body
    # And the fields exist, so this test fails loudly if the shape changes.
    assert set(asdict(item)) == {"id", "title", "text", "verdict", "score", "top_comments"}


def test_frozen_sample_loads_in_order_and_refuses_to_come_up_short(tmp_path):
    import json
    import pytest
    from jevbench.data import load

    items = [Item(id=str(i), title="a", text="b", verdict="nta", score=0, top_comments=[])
             for i in range(3)]
    path = tmp_path / "sample.jsonl"
    path.write_text("".join(json.dumps(asdict(i)) + "\n" for i in items))
    assert [i.id for i in load(path, 2)] == ["0", "1"]
    assert len(load(path)) == 3
    with pytest.raises(ValueError, match="fewer"):
        load(path, 4)
