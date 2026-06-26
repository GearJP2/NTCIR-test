import csv
import json

import pytest

from app.schemas.event import EventKind, EventRecord
from services.events.manifest import EventManifestError, write_event_manifest
from services.events.semantic_chunking_report import (
    assert_semantic_chunking_ready,
    build_semantic_chunking_report,
    write_semantic_chunking_report_json,
    write_semantic_chunking_report_markdown,
)


def event_record(**overrides) -> EventRecord:
    values = {
        "schema_version": "1.0",
        "processing_version": "test",
        "event_id": "day1_Allie_08_M_VISUAL_TEXT_00001",
        "participant_id": "Allie",
        "video_id": "day1_Allie_08",
        "event_kind": EventKind.SEMANTIC_MACRO,
        "start_ms": 400_000,
        "end_ms": 460_000,
        "duration_ms": 60_000,
        "video_uri": "hf://castle/08.mp4",
        "transcript": "macro transcript",
        "coverage": {
            "video": True,
            "transcript": True,
            "heart_rate": False,
            "gaze": False,
            "thermal": False,
        },
    }
    values.update(overrides)
    return EventRecord.model_validate(values)


def test_semantic_chunking_report_marks_valid_manifest_ready(tmp_path):
    manifest = tmp_path / "events.jsonl"
    sweep = tmp_path / "sweep.csv"
    violations = tmp_path / "violations.csv"
    macro = event_record()
    micro = event_record(
        event_id="day1_Allie_08_E_VISUAL_TEXT_00001",
        event_kind=EventKind.SEMANTIC_MICRO,
        parent_event_id=macro.event_id,
        start_ms=400_000,
        end_ms=430_000,
        duration_ms=30_000,
    )
    write_event_manifest(manifest, [macro, micro])
    write_sweep(sweep)
    write_violations(violations)

    report = build_semantic_chunking_report(
        manifest_path=manifest,
        transcript_weight=0.25,
        sweep_summary_path=sweep,
        modality_violations_path=violations,
    )

    assert report.chunking_status == "ready"
    assert report.micro_event_count == 1
    assert report.selected_weight_boundary_f1 == 0.6
    assert_semantic_chunking_ready(report)


def test_semantic_chunking_report_blocks_duration_violations(tmp_path):
    manifest = tmp_path / "events.jsonl"
    sweep = tmp_path / "sweep.csv"
    violations = tmp_path / "violations.csv"
    macro = event_record()
    micro = event_record(
        event_id="day1_Allie_08_E_VISUAL_TEXT_00001",
        event_kind=EventKind.SEMANTIC_MICRO,
        parent_event_id=macro.event_id,
        start_ms=400_000,
        end_ms=405_000,
        duration_ms=5_000,
    )
    write_event_manifest(manifest, [macro, micro])
    write_sweep(sweep)
    write_violations(violations)

    report = build_semantic_chunking_report(
        manifest_path=manifest,
        transcript_weight=0.25,
        sweep_summary_path=sweep,
        modality_violations_path=violations,
    )

    assert report.chunking_status == "blocked"
    with pytest.raises(EventManifestError, match="shorter"):
        assert_semantic_chunking_ready(report)


def test_semantic_chunking_report_writers_emit_markdown_and_json(tmp_path):
    manifest = tmp_path / "events.jsonl"
    sweep = tmp_path / "sweep.csv"
    violations = tmp_path / "violations.csv"
    macro = event_record()
    micro = event_record(
        event_id="day1_Allie_08_E_VISUAL_TEXT_00001",
        event_kind=EventKind.SEMANTIC_MICRO,
        parent_event_id=macro.event_id,
        start_ms=400_000,
        end_ms=430_000,
        duration_ms=30_000,
    )
    write_event_manifest(manifest, [macro, micro])
    write_sweep(sweep)
    write_violations(violations)
    report = build_semantic_chunking_report(
        manifest_path=manifest,
        transcript_weight=0.25,
        sweep_summary_path=sweep,
        modality_violations_path=violations,
    )
    markdown = tmp_path / "report.md"
    json_path = tmp_path / "report.json"

    write_semantic_chunking_report_markdown(markdown, report)
    write_semantic_chunking_report_json(json_path, report)

    assert "Status: `ready`" in markdown.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["micro_event_count"] == 1


def write_sweep(path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "transcript_weight",
                "boundary_f1_micro",
                "retrieval_recall_at_1",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "transcript_weight": "0.25",
                "boundary_f1_micro": "0.6",
                "retrieval_recall_at_1": "0.333",
            }
        )


def write_violations(path):
    path.write_text(
        "event_id,participant_id,day,video_id,modality,readiness_status,blocker\n",
        encoding="utf-8",
    )
