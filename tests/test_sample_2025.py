import pytest

from jevbench import sample_2025


def test_2025_sample_is_frozen_by_seed_and_filters_leaks(monkeypatch):
    monkeypatch.setattr(sample_2025, "TARGET", {v: 1 for v in sample_2025.FLAIR.values()})
    rows = []
    for i, (flair, verdict) in enumerate(sample_2025.FLAIR.items()):
        for j in range(2):
            rows.append({"id": f"{verdict}{j}", "title": "AITA for this?",
                         "selftext_cleaned": "ordinary story " * 40,
                         "link_flair_text": flair, "score": 0,
                         "created_utc": "2025-02-01", "n_verdicts": 30,
                         **{f"comments_prop_{v}": 0.2
                            for v in ("NTA", "YTA", "ESH", "NAH", "INFO")}})
    rows[0]["selftext_cleaned"] += " EDIT: I'm the asshole"
    one = sample_2025.select(rows, 7)
    two = sample_2025.select(rows, 7)
    assert one == two
    items, labels, eligible = one
    assert len(items) == len(labels) == 4
    assert eligible["nta"] == 1
    assert all(item.id != "nta0" and item.top_comments == [] for item in items)


def test_fetched_rows_must_match_the_pinned_hash(monkeypatch, tmp_path):
    from jevbench import sample_2025
    monkeypatch.setattr(sample_2025, "get_json",
                        lambda url: {"num_rows_total": 1, "rows": [{"row": {"id": "x"}}]})
    out = tmp_path / "raw.jsonl"
    with pytest.raises(SystemExit, match="has changed"):
        sample_2025.fetch_raw(out)
    assert not out.exists()
