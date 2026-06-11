from pathlib import Path
import json
from uuid import UUID

import pytest

from scripts.ingest_corpus import (
    _ingest_all,
    has_indexed_modalities,
    media_files_from_directory,
    media_files_from_manifest,
    media_files_matching_ids,
    media_files_without_indexed_modalities,
    media_files_starting_at,
    media_id_for_path,
    normalize_modalities,
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


def test_normalize_modalities_defaults_to_all_supported_modalities():
    assert normalize_modalities(None) == {"visual", "audio", "asr"}


def test_normalize_modalities_accepts_repeated_selected_modalities():
    assert normalize_modalities(["visual", "asr"]) == {"visual", "asr"}


def test_normalize_modalities_rejects_unknown_modality():
    with pytest.raises(Exception):
        normalize_modalities(["caption"])


class FakeMilvusClient:
    def __init__(self, indexed: dict[tuple[str, str], bool]):
        self.indexed = indexed

    def query(self, collection_name, filter, output_fields, limit):
        media_id = filter.split('"')[1]
        if self.indexed.get((collection_name, media_id), False):
            return [{"media_id": media_id}]
        return []


def test_has_indexed_modalities_requires_all_selected_modalities():
    client = FakeMilvusClient(
        {
            ("visual_keyframes", "v_1"): True,
            ("audio_segments", "v_1"): False,
        }
    )

    assert has_indexed_modalities("v_1", {"visual"}, client) is True
    assert has_indexed_modalities("v_1", {"visual", "audio"}, client) is False


def test_media_files_without_indexed_modalities_keeps_only_missing_media():
    files = [Path("v_1.mp4"), Path("v_2.mp4")]
    client = FakeMilvusClient(
        {
            ("visual_keyframes", "v_1"): True,
            ("visual_keyframes", "v_2"): False,
        }
    )

    assert media_files_without_indexed_modalities(
        files,
        media_id_source="filename",
        modalities={"visual"},
        milvus_client=client,
    ) == [Path("v_2.mp4")]


@pytest.mark.asyncio
async def test_ingest_all_passes_modalities_and_keyframe_interval(monkeypatch, tmp_path):
    calls = []

    async def fake_run_ingestion_pipeline(
        asset,
        local_path,
        modalities=None,
        keyframe_interval_sec=None,
    ):
        calls.append(
            {
                "media_id": asset.media_id,
                "local_path": local_path,
                "modalities": modalities,
                "keyframe_interval_sec": keyframe_interval_sec,
            }
        )

    monkeypatch.setattr(
        "services.ingestion.pipeline.run_ingestion_pipeline",
        fake_run_ingestion_pipeline,
    )
    video = tmp_path / "v_123.mp4"
    video.write_bytes(b"video")

    await _ingest_all(
        [video],
        language="en",
        max_concurrency=1,
        media_id_source="filename",
        modalities={"visual"},
        keyframe_interval_sec=10.0,
    )

    assert calls == [
        {
            "media_id": "v_123",
            "local_path": video,
            "modalities": {"visual"},
            "keyframe_interval_sec": 10.0,
        }
    ]
