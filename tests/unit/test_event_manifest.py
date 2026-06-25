import json

import pytest
from pydantic import ValidationError

from app.schemas.event import EventKind, EventRecord
from services.events.manifest import (
    EventManifestError,
    load_event_manifest,
    validate_event_manifest,
    write_event_manifest,
)


def event_record(**overrides) -> EventRecord:
    values = {
        "schema_version": "1.0",
        "processing_version": "test",
        "event_id": "P01_V01_M001",
        "participant_id": "P01",
        "video_id": "V01",
        "event_kind": EventKind.SEMANTIC_MACRO,
        "start_ms": 0,
        "end_ms": 120_000,
        "duration_ms": 120_000,
        "boundary_confidence": 0.8,
        "video_uri": "file:///castle/V01.mp4",
        "coverage": {
            "video": True,
            "transcript": False,
            "heart_rate": False,
            "gaze": False,
            "thermal": False,
        },
    }
    values.update(overrides)
    return EventRecord.model_validate(values)


def test_event_record_rejects_incorrect_duration():
    with pytest.raises(ValidationError, match="duration_ms"):
        event_record(duration_ms=119_000)


def test_event_record_keeps_core_interval_separate_from_retrieval_context():
    record = event_record(
        retrieval_context_start_ms=0,
        retrieval_context_end_ms=125_000,
    )

    assert (record.start_ms, record.end_ms) == (0, 120_000)
    assert record.retrieval_context_end_ms == 125_000


def test_event_record_requires_coverage_to_match_attached_evidence():
    with pytest.raises(ValidationError, match="coverage.heart_rate"):
        event_record(
            heart_rate={"mean_bpm": 90, "valid_ratio": 1.0},
        )


def test_manifest_validates_micro_event_parent():
    macro = event_record()
    micro = event_record(
        event_id="P01_V01_E001",
        event_kind=EventKind.SEMANTIC_MICRO,
        parent_event_id=macro.event_id,
        start_ms=10_000,
        end_ms=30_000,
        duration_ms=20_000,
    )

    validate_event_manifest([macro, micro])


def test_manifest_rejects_overlapping_semantic_events():
    first = event_record(
        event_id="P01_V01_M001",
        start_ms=0,
        end_ms=60_000,
        duration_ms=60_000,
    )
    second = event_record(
        event_id="P01_V01_M002",
        start_ms=50_000,
        end_ms=100_000,
        duration_ms=50_000,
    )

    with pytest.raises(EventManifestError, match="overlapping"):
        validate_event_manifest([first, second])


def test_event_manifest_round_trip(tmp_path):
    path = tmp_path / "events.jsonl"
    records = [event_record()]

    write_event_manifest(path, records)

    assert load_event_manifest(path) == records


def test_load_event_manifest_reports_line_number(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"event_id": "invalid"}) + "\n", encoding="utf-8")

    with pytest.raises(EventManifestError, match=r"events.jsonl:1:"):
        load_event_manifest(path)
