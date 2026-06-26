import csv
import json

from typer.testing import CliRunner

from app.schemas.event import EventKind, EventRecord
from scripts.enrich_castle_heart_rate import app
from services.events.manifest import load_event_manifest


def test_enrich_castle_heart_rate_cli_writes_manifest_and_summary(tmp_path):
    input_manifest = tmp_path / "events.jsonl"
    output_manifest = tmp_path / "events_hr.jsonl"
    output_summary = tmp_path / "hr_summary.csv"
    heart_rate_csv = tmp_path / "day1.csv"
    inventory = tmp_path / "inventory.csv"

    event = EventRecord.model_validate(
        {
            "schema_version": "1.0",
            "processing_version": "test",
            "event_id": "day1_Allie_08_E0001",
            "participant_id": "Allie",
            "video_id": "day1_Allie_08",
            "event_kind": EventKind.FIXED_120S,
            "start_ms": 10_000,
            "end_ms": 20_000,
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
    )
    input_manifest.write_text(
        json.dumps(event.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    heart_rate_csv.write_text(
        "time,bpm,confidence\n"
        "00:01:50.0,60,3\n"
        "00:01:55.0,64,3\n",
        encoding="utf-8",
    )
    inventory.write_text(
        "source,path,time_field,original_time_format,first_time,last_time,"
        "first_offset_ms,last_offset_ms,row_count,anchor_status,notes\n"
        f"day1/Allie/08.ACCL,{tmp_path / '08.ACCL.csv'},time,clock,"
        "00:01:40.000,00:59:59.000,100000,3599000,10,status,note\n"
        f"auxiliary/heartrate/Allie/day1,{heart_rate_csv},time,elapsed,"
        "00:00:02.0,23:59:59.0,2000,86399000,10,status,note\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            str(input_manifest),
            "--output-manifest",
            str(output_manifest),
            "--timeline-inventory",
            str(inventory),
            "--output-summary",
            str(output_summary),
            "--day",
            "day1",
            "--participant-id",
            "Allie",
        ],
    )

    assert result.exit_code == 0, result.output
    enriched = load_event_manifest(output_manifest)
    assert enriched[0].coverage.heart_rate is True
    assert enriched[0].heart_rate is not None
    assert enriched[0].heart_rate.mean_bpm == 62

    with output_summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["clock_start_ms"] == "110000"
    assert rows[0]["clock_end_ms"] == "120000"
    assert rows[0]["overlapping_samples"] == "2"
    assert rows[0]["valid_samples"] == "2"


def test_enrich_castle_heart_rate_cli_reports_missing_source(tmp_path):
    input_manifest = tmp_path / "events.jsonl"
    output_manifest = tmp_path / "events_hr.jsonl"
    inventory = tmp_path / "inventory.csv"
    event = EventRecord.model_validate(
        {
            "schema_version": "1.0",
            "processing_version": "test",
            "event_id": "day1_Allie_08_E0001",
            "participant_id": "Allie",
            "video_id": "day1_Allie_08",
            "event_kind": EventKind.FIXED_120S,
            "start_ms": 10_000,
            "end_ms": 20_000,
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
    )
    input_manifest.write_text(
        json.dumps(event.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    inventory.write_text(
        "source,path,time_field,original_time_format,first_time,last_time,"
        "first_offset_ms,last_offset_ms,row_count,anchor_status,notes\n"
        f"day1/Allie/08.ACCL,{tmp_path / '08.ACCL.csv'},time,clock,"
        "00:01:40.000,00:59:59.000,100000,3599000,10,status,note\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            str(input_manifest),
            "--output-manifest",
            str(output_manifest),
            "--timeline-inventory",
            str(inventory),
            "--day",
            "day1",
            "--participant-id",
            "Allie",
        ],
    )

    assert result.exit_code != 0
    assert "no heart-rate source for Allie/day1" in result.output
