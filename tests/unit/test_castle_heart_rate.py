import pytest

from app.schemas.event import EventKind, EventRecord
from services.dataset.castle_heart_rate import (
    HeartRateSample,
    attach_heart_rate_to_events,
    load_heart_rate_samples,
    load_heart_rate_sources,
    load_recording_clock_starts,
    summarize_heart_rate,
    summarize_heart_rate_enrichment,
    write_heart_rate_enrichment_summary,
)


def event_record(**overrides) -> EventRecord:
    values = {
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
    values.update(overrides)
    return EventRecord.model_validate(values)


def test_load_heart_rate_samples_parses_elapsed_rows(tmp_path):
    path = tmp_path / "hr.csv"
    path.write_text(
        "time,bpm,confidence\n"
        "00:00:02.0,64,3\n"
        "00:00:04.5,65,2\n",
        encoding="utf-8",
    )

    assert load_heart_rate_samples(path) == [
        HeartRateSample(offset_ms=2_000, bpm=64.0, confidence=3.0),
        HeartRateSample(offset_ms=4_500, bpm=65.0, confidence=2.0),
    ]


def test_summarize_heart_rate_filters_invalid_samples_and_computes_stats():
    summary = summarize_heart_rate(
        [
            HeartRateSample(offset_ms=1_000, bpm=60, confidence=3),
            HeartRateSample(offset_ms=2_000, bpm=66, confidence=3),
            HeartRateSample(offset_ms=3_000, bpm=250, confidence=3),
            HeartRateSample(offset_ms=4_000, bpm=80, confidence=0),
        ],
        start_ms=0,
        end_ms=5_000,
        baseline_bpm=61,
    )

    assert summary is not None
    assert summary.mean_bpm == 63
    assert summary.min_bpm == 60
    assert summary.max_bpm == 66
    assert summary.std_bpm == 3
    assert summary.slope_bpm_s == pytest.approx(6)
    assert summary.baseline_delta == 2
    assert summary.valid_ratio == 0.5


def test_inventory_loaders_map_sources(tmp_path):
    hr_path = tmp_path / "day1.csv"
    inventory = tmp_path / "inventory.csv"
    inventory.write_text(
        "source,path,time_field,original_time_format,first_time,last_time,"
        "first_offset_ms,last_offset_ms,row_count,anchor_status,notes\n"
        f"day1/Allie/08.ACCL,{tmp_path / '08.ACCL.csv'},time,clock,"
        "08:05:17.180,08:59:59.998,29117180,32399998,10,status,note\n"
        f"auxiliary/heartrate/Allie/day1,{hr_path},time,elapsed,"
        "00:00:02.0,23:59:59.0,2000,86399000,10,status,note\n",
        encoding="utf-8",
    )

    assert load_recording_clock_starts(inventory) == {
        "day1_Allie_08": 29_117_180
    }
    assert load_heart_rate_sources(inventory)[("day1", "Allie")].path == hr_path


def test_attach_heart_rate_converts_event_relative_time_to_clock_offset():
    event = event_record()
    enriched = attach_heart_rate_to_events(
        [event],
        [
            HeartRateSample(offset_ms=110_000, bpm=60, confidence=3),
            HeartRateSample(offset_ms=115_000, bpm=64, confidence=3),
        ],
        recording_clock_starts_ms={"day1_Allie_08": 100_000},
        source_uri="auxiliary/heartrate/Allie/day1",
    )

    assert enriched[0].coverage.heart_rate is True
    assert enriched[0].heart_rate is not None
    assert enriched[0].heart_rate.mean_bpm == 62
    assert enriched[0].raw_evidence_uris["heart_rate"] == [
        "auxiliary/heartrate/Allie/day1"
    ]


def test_summarize_heart_rate_enrichment_reports_mapping_and_sample_counts(tmp_path):
    event = event_record()
    rows = summarize_heart_rate_enrichment(
        [event],
        [
            HeartRateSample(offset_ms=110_000, bpm=60, confidence=3),
            HeartRateSample(offset_ms=115_000, bpm=250, confidence=3),
        ],
        recording_clock_starts_ms={"day1_Allie_08": 100_000},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.clock_start_ms == 110_000
    assert row.clock_end_ms == 120_000
    assert row.overlapping_samples == 2
    assert row.valid_samples == 1
    assert row.mean_bpm == 60
    assert row.valid_ratio == 0.5

    output = tmp_path / "summary.csv"
    write_heart_rate_enrichment_summary(output, rows)

    assert "event_id,video_id,event_start_ms" in output.read_text(encoding="utf-8")
