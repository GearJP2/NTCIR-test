import json

import pytest

from evaluation.manifest import iter_evaluation_queries, load_evaluation_manifest


def test_load_evaluation_manifest_jsonl(tmp_path):
    manifest = tmp_path / "activitynet_dev50.jsonl"
    row = {
        "media_id": "v_123",
        "video_path": "videos/v_123.mp4",
        "duration_sec": 120.4,
        "queries": [
            {
                "query_id": "v_123:0",
                "query": "A woman is doing sit ups",
                "ground_truth": {"start_sec": 39.8, "end_sec": 54.6},
            }
        ],
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    videos = load_evaluation_manifest(manifest)
    queries = iter_evaluation_queries(videos)

    assert len(videos) == 1
    assert videos[0].media_id == "v_123"
    assert videos[0].video_path == tmp_path / "videos/v_123.mp4"
    assert videos[0].duration_sec == 120.4
    assert len(queries) == 1
    assert queries[0].query_id == "v_123:0"
    assert queries[0].ground_truth.start_sec == 39.8
    assert queries[0].ground_truth.end_sec == 54.6


def test_load_evaluation_manifest_defaults_query_id(tmp_path):
    manifest = tmp_path / "activitynet_dev50.jsonl"
    row = {
        "media_id": "v_123",
        "video_path": "/data/v_123.mp4",
        "queries": [
            {
                "query": "A woman is doing sit ups",
                "ground_truth": {"start_sec": 39.8, "end_sec": 54.6},
            }
        ],
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    query = iter_evaluation_queries(load_evaluation_manifest(manifest))[0]

    assert query.query_id == "v_123:0"


def test_load_evaluation_manifest_rejects_invalid_interval(tmp_path):
    manifest = tmp_path / "bad.jsonl"
    row = {
        "media_id": "v_123",
        "video_path": "videos/v_123.mp4",
        "queries": [
            {
                "query": "A woman is doing sit ups",
                "ground_truth": {"start_sec": 54.6, "end_sec": 39.8},
            }
        ],
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-positive ground-truth duration"):
        load_evaluation_manifest(manifest)
