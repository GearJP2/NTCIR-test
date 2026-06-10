import json

import pytest

from evaluation.index_diagnostics import (
    diagnostics_to_json,
    inspect_media_index,
    list_indexed_media_candidates,
)


class FakeMilvus:
    def __init__(self):
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        collection = kwargs["collection_name"]
        if collection == "visual_keyframes":
            return [{"frame_id": "frame-1", "media_id": "v_123", "timestamp_sec": 10.0}]
        if collection == "text_transcripts":
            return []
        if collection == "audio_segments":
            return [{"segment_id": "audio-1", "media_id": "v_123", "start_sec": 0.0}]
        raise AssertionError(f"unexpected collection {collection}")


class FailingMilvus:
    def query(self, **kwargs):
        raise RuntimeError(f"{kwargs['collection_name']} unavailable")


def test_inspect_media_index_reports_collection_readiness():
    client = FakeMilvus()

    report = inspect_media_index("v_123", client, sample_limit=5)

    assert report["media_id"] == "v_123"
    assert report["sample_limit"] == 5
    assert report["ready"] is True

    by_collection = {item["collection_name"]: item for item in report["collections"]}
    assert by_collection["visual_keyframes"]["ready"] is True
    assert by_collection["visual_keyframes"]["sample_count"] == 1
    assert by_collection["text_transcripts"]["ready"] is False
    assert by_collection["audio_segments"]["ready"] is True

    assert client.calls[0]["filter"] == 'media_id == "v_123"'
    assert client.calls[0]["limit"] == 5


def test_inspect_media_index_captures_collection_errors():
    report = inspect_media_index("v_123", FailingMilvus())

    assert report["ready"] is False
    assert all(item["error"] for item in report["collections"])


def test_inspect_media_index_rejects_non_positive_sample_limit():
    with pytest.raises(ValueError, match="sample_limit must be positive"):
        inspect_media_index("v_123", FakeMilvus(), sample_limit=0)


def test_diagnostics_to_json_serializes_report():
    report = inspect_media_index("v_123", FakeMilvus(), sample_limit=1)

    data = json.loads(diagnostics_to_json(report))

    assert data["media_id"] == "v_123"
    assert len(data["collections"]) == 3


def test_list_indexed_media_candidates_counts_sampled_media():
    candidates = list_indexed_media_candidates(FakeMilvus(), sample_limit=10)

    assert candidates == [
        {
            "media_id": "v_123",
            "visual": 1,
            "asr": 0,
            "audio": 1,
            "total": 2,
        }
    ]


def test_list_indexed_media_candidates_rejects_non_positive_sample_limit():
    with pytest.raises(ValueError, match="sample_limit must be positive"):
        list_indexed_media_candidates(FakeMilvus(), sample_limit=0)
