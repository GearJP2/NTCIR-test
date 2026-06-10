from scripts.download_activitynet_dev import (
    make_video_paths_relative_to_manifest,
    select_rows,
    video_filename,
    youtube_id_from_activitynet_video_id,
)


def test_youtube_id_from_activitynet_video_id_strips_prefix():
    assert youtube_id_from_activitynet_video_id("v_uqiMw7tQ1Cc") == "uqiMw7tQ1Cc"
    assert youtube_id_from_activitynet_video_id("uqiMw7tQ1Cc") == "uqiMw7tQ1Cc"


def test_video_filename_uses_activitynet_id():
    assert video_filename("v_uqiMw7tQ1Cc") == "v_uqiMw7tQ1Cc.mp4"


def test_select_rows_is_seeded():
    rows = [{"video_id": str(i)} for i in range(5)]

    assert select_rows(rows, seed=19) == select_rows(rows, seed=19)
    assert select_rows(rows, seed=19) != rows


def test_make_video_paths_relative_to_manifest(tmp_path):
    video_path = tmp_path / "data" / "activitynet" / "videos" / "v_123.mp4"
    manifest_path = tmp_path / "data" / "manifests" / "activitynet_dev50.jsonl"
    rows = [{"media_id": "v_123", "video_path": str(video_path), "queries": []}]

    normalized = make_video_paths_relative_to_manifest(rows, manifest_path)

    assert normalized[0]["video_path"] == "../activitynet/videos/v_123.mp4"
