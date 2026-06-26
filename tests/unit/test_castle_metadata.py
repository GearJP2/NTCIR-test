from services.dataset.castle_metadata import (
    inspect_gaze_csv,
    inspect_heart_rate_csv,
    inspect_main_metadata_csv,
    parse_clock_time_ms,
    parse_elapsed_time_ms,
    parse_gaze_session_start,
    parse_seconds_ms,
)


def test_parse_clock_time_ms_supports_millisecond_fraction():
    assert parse_clock_time_ms("08:05:17.180") == 29_117_180


def test_parse_elapsed_time_ms_reuses_hh_mm_ss_fraction():
    assert parse_elapsed_time_ms("00:00:02.0") == 2_000


def test_parse_gaze_session_start_extracts_header_anchor():
    assert (
        parse_gaze_session_start("TIME(2024/12/04 16:54:18.272)")
        == "2024/12/04 16:54:18.272"
    )


def test_parse_seconds_ms_rounds_decimal_seconds():
    assert parse_seconds_ms("0.00781") == 8


def test_inspect_main_metadata_csv_reports_clock_range(tmp_path):
    path = tmp_path / "08.ACCL.csv"
    path.write_text(
        "time, x, y, z\n"
        "08:05:17.180, 1, 2, 3\n"
        "08:05:18.180, 4, 5, 6\n",
        encoding="utf-8",
    )

    inspection = inspect_main_metadata_csv(path, source="day1/Allie/08.ACCL")

    assert inspection.row_count == 2
    assert inspection.first_offset_ms == 29_117_180
    assert inspection.last_offset_ms == 29_118_180
    assert inspection.anchor_status == "date/timezone unresolved"


def test_inspect_heart_rate_csv_reports_elapsed_range(tmp_path):
    path = tmp_path / "day1.csv"
    path.write_text(
        "time,bpm,confidence\n"
        "00:00:02.0,64,3\n"
        "00:00:04.0,64,3\n",
        encoding="utf-8",
    )

    inspection = inspect_heart_rate_csv(path, source="hr")

    assert inspection.first_offset_ms == 2_000
    assert inspection.last_offset_ms == 4_000
    assert inspection.anchor_status == "participant/day session start unresolved"


def test_inspect_gaze_csv_reports_header_session_start(tmp_path):
    path = tmp_path / "Allie.csv"
    path.write_text(
        "MEDIA_ID,TIME(2024/12/04 16:54:18.272),FPOGX\n"
        "0,0.00000,0.5\n"
        "0,0.00781,0.6\n",
        encoding="utf-8",
    )

    inspection = inspect_gaze_csv(path, source="gaze")

    assert inspection.first_offset_ms == 0
    assert inspection.last_offset_ms == 8
    assert "2024/12/04 16:54:18.272" in inspection.notes
