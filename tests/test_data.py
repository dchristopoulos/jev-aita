"""The model must reach its own verdict, not read one off the page."""

from dataclasses import asdict

from jevbench.data import Item, leaks_verdict


def test_verdict_edits_are_detected():
    assert leaks_verdict("EDIT: ok everyone, seems like I'm NTA here")
    assert leaks_verdict("so am i yta for this?")
    assert leaks_verdict("verdict was ESH apparently")


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
