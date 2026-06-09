import json

from evaluation.activitynet_manifest import (
    build_activitynet_manifest_rows,
    write_manifest_jsonl,
)
from evaluation.manifest import load_evaluation_manifest


def test_build_activitynet_manifest_rows_from_sentences_and_timestamps(tmp_path):
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "v_123.mp4").write_bytes(b"fake")
    rows = [
        {
            "video_id": "v_123",
            "duration": 120.0,
            "sentences": ["A woman is doing sit ups", "A woman rests"],
            "timestamps": [[39.8, 54.6], [55.0, 60.0]],
        }
    ]

    manifest_rows = build_activitynet_manifest_rows(rows, video_root=video_root)

    assert len(manifest_rows) == 1
    assert manifest_rows[0]["media_id"] == "v_123"
    assert manifest_rows[0]["video_path"] == str(video_root / "v_123.mp4")
    assert manifest_rows[0]["queries"][0] == {
        "query_id": "v_123:0",
        "query": "A woman is doing sit ups",
        "ground_truth": {"start_sec": 39.8, "end_sec": 54.6},
    }


def test_build_activitynet_manifest_rows_filters_missing_video_by_default(tmp_path):
    rows = [
        {
            "video_id": "v_missing",
            "sentences": ["A woman is doing sit ups"],
            "timestamps": [[39.8, 54.6]],
        }
    ]

    manifest_rows = build_activitynet_manifest_rows(rows, video_root=tmp_path)

    assert manifest_rows == []


def test_build_activitynet_manifest_rows_can_allow_missing_video(tmp_path):
    rows = [
        {
            "video_id": "v_missing",
            "sentences": ["A woman is doing sit ups"],
            "timestamps": [[39.8, 54.6]],
        }
    ]

    manifest_rows = build_activitynet_manifest_rows(
        rows,
        video_root=tmp_path,
        require_video_file=False,
    )

    assert len(manifest_rows) == 1
    assert manifest_rows[0]["video_path"] == str(tmp_path / "v_missing.mp4")


def test_build_activitynet_manifest_rows_respects_query_cap(tmp_path):
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "v_1.mp4").write_bytes(b"fake")
    rows = [
        {
            "video_id": "v_1",
            "sentences": ["one", "two", "three"],
            "timestamps": [[0, 1], [2, 3], [4, 5]],
        }
    ]

    manifest_rows = build_activitynet_manifest_rows(
        rows,
        video_root=video_root,
        max_queries=2,
    )

    assert len(manifest_rows[0]["queries"]) == 2


def test_write_manifest_jsonl_round_trips_with_loader(tmp_path):
    output_path = tmp_path / "manifest.jsonl"
    rows = [
        {
            "media_id": "v_123",
            "video_path": "videos/v_123.mp4",
            "duration_sec": 120.0,
            "queries": [
                {
                    "query_id": "v_123:0",
                    "query": "A woman is doing sit ups",
                    "ground_truth": {"start_sec": 39.8, "end_sec": 54.6},
                }
            ],
        }
    ]

    write_manifest_jsonl(rows, output_path)
    raw = output_path.read_text(encoding="utf-8").strip()
    assert json.loads(raw)["media_id"] == "v_123"

    videos = load_evaluation_manifest(output_path)
    assert videos[0].queries[0].query == "A woman is doing sit ups"
