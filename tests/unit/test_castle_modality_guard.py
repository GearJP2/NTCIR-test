import csv
import json

from typer.testing import CliRunner

from app.schemas.event import EventKind, EventRecord
from scripts.check_castle_manifest_modality_readiness import app
from services.dataset.castle_modality_guard import (
    find_blocked_modality_violations,
    load_modality_readiness,
    write_modality_violations,
)


def event_record(**overrides) -> EventRecord:
    values = {
        "schema_version": "1.0",
        "processing_version": "test",
        "event_id": "day1_Allie_08_E0001",
        "participant_id": "Allie",
        "video_id": "day1_Allie_08",
        "event_kind": EventKind.FIXED_120S,
        "start_ms": 0,
        "end_ms": 10_000,
        "duration_ms": 10_000,
        "video_uri": "hf://castle/08.mp4",
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


def write_readiness(path):
    path.write_text(
        "participant_id,day,modality,attach_to_event_records,status,evidence_rows,"
        "blocker,next_action\n"
        "Allie,day1,heart_rate,True,attachable_with_clock_day_join,2,,ok\n"
        "Allie,day1,gaze,False,blocked_no_clock_overlap,6,no overlap,find mapping\n"
        "Allie,day1,thermal,False,blocked_unassigned,39,unassigned,find manifest\n",
        encoding="utf-8",
    )


def test_find_blocked_modality_violations_rejects_blocked_gaze(tmp_path):
    readiness_path = tmp_path / "readiness.csv"
    write_readiness(readiness_path)
    event = event_record(
        coverage={
            "video": True,
            "transcript": False,
            "heart_rate": False,
            "gaze": True,
            "thermal": False,
        },
        gaze={
            "valid_fixation_count": 1,
            "valid_ratio": 1.0,
        },
    )

    violations = find_blocked_modality_violations(
        [event],
        load_modality_readiness(readiness_path),
    )

    assert len(violations) == 1
    assert violations[0].modality == "gaze"
    assert violations[0].readiness_status == "blocked_no_clock_overlap"


def test_write_modality_violations_emits_header(tmp_path):
    output = tmp_path / "violations.csv"

    write_modality_violations(output, [])

    assert "event_id,participant_id,day" in output.read_text(encoding="utf-8")


def test_check_manifest_modality_readiness_cli_fails_for_blocked_gaze(tmp_path):
    manifest = tmp_path / "events.jsonl"
    readiness = tmp_path / "readiness.csv"
    violations = tmp_path / "violations.csv"
    write_readiness(readiness)
    event = event_record(
        coverage={
            "video": True,
            "transcript": False,
            "heart_rate": False,
            "gaze": True,
            "thermal": False,
        },
        gaze={
            "valid_fixation_count": 1,
            "valid_ratio": 1.0,
        },
    )
    manifest.write_text(
        json.dumps(event.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            str(manifest),
            "--modality-readiness",
            str(readiness),
            "--output-violations",
            str(violations),
        ],
    )

    assert result.exit_code == 1
    with violations.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["modality"] == "gaze"


def test_check_manifest_modality_readiness_cli_passes_for_video_only(tmp_path):
    manifest = tmp_path / "events.jsonl"
    readiness = tmp_path / "readiness.csv"
    violations = tmp_path / "violations.csv"
    write_readiness(readiness)
    manifest.write_text(
        json.dumps(event_record().model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            str(manifest),
            "--modality-readiness",
            str(readiness),
            "--output-violations",
            str(violations),
        ],
    )

    assert result.exit_code == 0
    assert list(csv.DictReader(violations.open(newline="", encoding="utf-8"))) == []
