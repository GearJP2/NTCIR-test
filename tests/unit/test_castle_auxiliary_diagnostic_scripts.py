import csv
import json

from typer.testing import CliRunner

from scripts.build_castle_gaze_alignment_diagnostics import app as gaze_app
from scripts.build_castle_modality_readiness import app as readiness_app
from scripts.build_castle_thermal_inventory import app as thermal_app


def test_gaze_alignment_diagnostics_cli_writes_stream_and_alignment_csvs(tmp_path):
    timeline = tmp_path / "timeline.csv"
    gaze_csv = tmp_path / "gaze.csv"
    streams = tmp_path / "streams.csv"
    alignment = tmp_path / "alignment.csv"
    timeline.write_text(
        "source,path,time_field,original_time_format,first_time,last_time,"
        "first_offset_ms,last_offset_ms,row_count,anchor_status,notes\n"
        "day1/Allie/08.ACCL,/tmp/08.ACCL.csv,time,clock,"
        "08:05:17.180,08:59:59.998,29117180,32399998,10,status,note\n",
        encoding="utf-8",
    )
    gaze_csv.write_text(
        "MEDIA_ID,MEDIA_NAME,TIME(2024/12/04 16:54:18.272),FPOGV,FPOGID,AOI,\n"
        "0,NewMedia0,0.00000,1,1,,\n"
        "0,NewMedia0,1.00000,1,2,,\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        gaze_app,
        [
            "--timeline-inventory",
            str(timeline),
            "--gaze-csv",
            str(gaze_csv),
            "--output-streams",
            str(streams),
            "--output-alignment",
            str(alignment),
        ],
    )

    assert result.exit_code == 0, result.output
    with streams.open(newline="", encoding="utf-8") as handle:
        stream_rows = list(csv.DictReader(handle))
    with alignment.open(newline="", encoding="utf-8") as handle:
        alignment_rows = list(csv.DictReader(handle))
    assert stream_rows[0]["media_name"] == "NewMedia0"
    assert alignment_rows[0]["recording_video_id"] == "day1_Allie_08"


def test_thermal_inventory_cli_accepts_offline_repository_file_jsonl(tmp_path):
    files_jsonl = tmp_path / "files.jsonl"
    output = tmp_path / "thermal.csv"
    files_jsonl.write_text(
        json.dumps(
            {
                "path": "auxiliary/thermal/IMG 0003.bmp",
                "size_bytes": 123,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        thermal_app,
        [
            "--repository-files-jsonl",
            str(files_jsonl),
            "--output-csv",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["assignment_status"] == "unassigned"
    assert rows[0]["timestamp_status"] == "no_timestamp_in_path"


def test_modality_readiness_cli_writes_readiness_gate(tmp_path):
    timeline = tmp_path / "timeline.csv"
    gaze = tmp_path / "gaze.csv"
    thermal = tmp_path / "thermal.csv"
    output = tmp_path / "readiness.csv"
    timeline.write_text(
        "source,path,time_field,original_time_format,first_time,last_time,"
        "first_offset_ms,last_offset_ms,row_count,anchor_status,notes\n"
        "day1/Allie/08.ACCL,/tmp/08.ACCL.csv,time,clock,"
        "08:05:17.180,08:59:59.998,29117180,32399998,10,status,note\n"
        "auxiliary/heartrate/Allie/day1,/tmp/day1.csv,time,elapsed,"
        "00:00:02.0,23:59:59.0,2000,86399000,10,status,note\n",
        encoding="utf-8",
    )
    gaze.write_text(
        "gaze_source,media_id,media_name,candidate_anchor,candidate_start_clock_ms,"
        "candidate_end_clock_ms,recording_video_id,recording_start_clock_ms,"
        "recording_end_clock_ms,overlap_ms,overlap_ratio,status,notes\n"
        "auxiliary/gaze/Allie,0,NewMedia0,header_clock_of_day,1,2,"
        "day1_Allie_08,29117180,32399998,0,0.0,no_overlap,note\n",
        encoding="utf-8",
    )
    thermal.write_text(
        "source,filename,size_bytes,sequence_index,inferred_day,"
        "inferred_participant_id,timestamp_status,assignment_status,notes\n"
        "auxiliary/thermal/IMG 0003.bmp,IMG 0003.bmp,123,3,,,"
        "no_timestamp_in_path,unassigned,note\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        readiness_app,
        [
            "--timeline-inventory",
            str(timeline),
            "--gaze-alignment",
            str(gaze),
            "--thermal-inventory",
            str(thermal),
            "--output-csv",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["modality"] for row in rows] == ["heart_rate", "gaze", "thermal"]
    assert rows[0]["attach_to_event_records"] == "True"
