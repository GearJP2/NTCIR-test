from pathlib import Path
import json
from uuid import UUID

import pytest

from scripts.ingest_corpus import (
    media_files_from_directory,
    media_files_from_manifest,
    media_files_matching_ids,
    media_files_starting_at,
    media_id_for_path,
)


def test_media_id_for_path_uses_filename_stem():
    assert media_id_for_path(Path("data/activitynet/videos/v_123.mp4"), "filename") == "v_123"


def test_media_id_for_path_uses_uuid():
    media_id = media_id_for_path(Path("video.mp4"), "uuid")

    assert str(UUID(media_id)) == media_id


def test_media_id_for_path_rejects_unknown_source():
    with pytest.raises(ValueError):
        media_id_for_path(Path("video.mp4"), "unknown")


def test_media_files_from_directory_skips_generated_artifacts(tmp_path):
    video = tmp_path / "v_123.mp4"
    audio = tmp_path / "v_123.audio.wav"
    keyframe_dir = tmp_path / "keyframes"
    keyframe_dir.mkdir()
    keyframe = keyframe_dir / "v_123_0.jpg"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    keyframe.write_bytes(b"image")

    assert media_files_from_directory(tmp_path) == [video]


def test_media_files_from_manifest_uses_manifest_video_paths(tmp_path):
    video = tmp_path / "videos" / "v_123.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "media_id": "v_123",
                "video_path": "videos/v_123.mp4",
                "duration_sec": 10,
                "queries": [
                    {
                        "query": "query",
                        "ground_truth": {"start_sec": 0, "end_sec": 1},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert media_files_from_manifest(manifest) == [video]


def test_media_files_starting_at_keeps_tail_from_media_id():
    files = [Path("v_1.mp4"), Path("v_2.mp4"), Path("v_3.mp4")]

    assert media_files_starting_at(files, "v_2") == [Path("v_2.mp4"), Path("v_3.mp4")]


def test_media_files_starting_at_rejects_unknown_media_id():
    with pytest.raises(ValueError):
        media_files_starting_at([Path("v_1.mp4")], "v_missing")


def test_media_files_matching_ids_keeps_manifest_order():
    files = [Path("v_1.mp4"), Path("v_2.mp4"), Path("v_3.mp4")]

    assert media_files_matching_ids(files, ["v_3", "v_1"]) == [
        Path("v_1.mp4"),
        Path("v_3.mp4"),
    ]


def test_media_files_matching_ids_rejects_unknown_media_id():
    with pytest.raises(ValueError):
        media_files_matching_ids([Path("v_1.mp4")], ["v_missing"])
