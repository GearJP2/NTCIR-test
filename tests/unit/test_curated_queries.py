import json
from pathlib import Path

import pytest

from evaluation.curated_queries import load_curated_queries


def test_load_castle_smoke_curated_queries():
    queries = load_curated_queries(Path("data/curated_queries/castle_smoke.jsonl"))

    assert len(queries) == 10
    assert queries[0].query_id == "castle_smoke:001"
    assert queries[0].query == "person eating a meal"
    assert "food" in queries[0].tags
    assert queries[-1].media_id is None


def test_load_curated_queries_accepts_optional_media_id(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "person using a laptop",
                "media_id": "castle_001",
                "tags": ["work"],
                "notes": "Pinned to one recording.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    query = load_curated_queries(path)[0]

    assert query.media_id == "castle_001"
    assert query.tags == ("work",)
    assert query.notes == "Pinned to one recording."


def test_load_curated_queries_rejects_short_query(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(
        json.dumps({"query_id": "q1", "query": "x", "tags": []}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least 3 characters"):
        load_curated_queries(path)


def test_load_curated_queries_rejects_non_list_tags(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(
        json.dumps({"query_id": "q1", "query": "person walking", "tags": "movement"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tags must be a list"):
        load_curated_queries(path)
