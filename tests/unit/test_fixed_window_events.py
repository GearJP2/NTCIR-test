import json

from typer.testing import CliRunner

from app.schemas.event import EventKind, RecordingRecord
from scripts.build_castle_fixed_manifest import app
from services.events.fixed_windows import build_fixed_window_events
from services.events.manifest import load_event_manifest


def recording(duration_ms: int) -> RecordingRecord:
    return RecordingRecord(
        participant_id="P01",
        video_id="V01",
        duration_ms=duration_ms,
        video_uri="file:///castle/V01.mp4",
    )


def test_build_30s_windows_uses_10s_overlap_and_keeps_final_partial_window():
    events = build_fixed_window_events(
        recording(65_000),
        event_kind=EventKind.FIXED_30S,
        processing_version="test",
    )

    assert [(event.start_ms, event.end_ms) for event in events] == [
        (0, 30_000),
        (20_000, 50_000),
        (40_000, 65_000),
        (60_000, 65_000),
    ]


def test_build_120s_windows_is_non_overlapping():
    events = build_fixed_window_events(
        recording(250_000),
        event_kind=EventKind.FIXED_120S,
        processing_version="test",
    )

    assert [(event.start_ms, event.end_ms) for event in events] == [
        (0, 120_000),
        (120_000, 240_000),
        (240_000, 250_000),
    ]


def test_build_fixed_manifest_cli(tmp_path):
    recordings_path = tmp_path / "recordings.jsonl"
    output_path = tmp_path / "fixed.jsonl"
    recordings_path.write_text(
        json.dumps(recording(45_000).model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            str(recordings_path),
            "--output-path",
            str(output_path),
            "--window",
            "30s",
            "--processing-version",
            "test",
        ],
    )

    assert result.exit_code == 0
    assert len(load_event_manifest(output_path)) == 3
