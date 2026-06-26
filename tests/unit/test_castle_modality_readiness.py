import csv

from services.dataset.castle_modality_readiness import (
    build_modality_readiness,
    write_modality_readiness,
)


def test_build_modality_readiness_marks_hr_attachable_and_gaze_thermal_blocked(
    tmp_path,
):
    timeline = tmp_path / "timeline.csv"
    gaze = tmp_path / "gaze.csv"
    thermal = tmp_path / "thermal.csv"
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

    rows = build_modality_readiness(
        participant_id="Allie",
        day="day1",
        timeline_inventory=timeline,
        gaze_alignment=gaze,
        thermal_inventory=thermal,
    )

    by_modality = {row.modality: row for row in rows}
    assert by_modality["heart_rate"].attach_to_event_records is True
    assert by_modality["gaze"].status == "blocked_no_clock_overlap"
    assert by_modality["thermal"].status == "blocked_unassigned"


def test_write_modality_readiness_round_trips_csv(tmp_path):
    timeline = tmp_path / "timeline.csv"
    gaze = tmp_path / "gaze.csv"
    thermal = tmp_path / "thermal.csv"
    timeline.write_text(
        "source,path,time_field,original_time_format,first_time,last_time,"
        "first_offset_ms,last_offset_ms,row_count,anchor_status,notes\n",
        encoding="utf-8",
    )
    gaze.write_text(
        "gaze_source,media_id,media_name,candidate_anchor,candidate_start_clock_ms,"
        "candidate_end_clock_ms,recording_video_id,recording_start_clock_ms,"
        "recording_end_clock_ms,overlap_ms,overlap_ratio,status,notes\n",
        encoding="utf-8",
    )
    thermal.write_text(
        "source,filename,size_bytes,sequence_index,inferred_day,"
        "inferred_participant_id,timestamp_status,assignment_status,notes\n",
        encoding="utf-8",
    )
    rows = build_modality_readiness(
        participant_id="Allie",
        day="day1",
        timeline_inventory=timeline,
        gaze_alignment=gaze,
        thermal_inventory=thermal,
    )
    output = tmp_path / "readiness.csv"

    write_modality_readiness(output, rows)

    with output.open(newline="", encoding="utf-8") as handle:
        parsed = list(csv.DictReader(handle))
    assert [row["modality"] for row in parsed] == ["heart_rate", "gaze", "thermal"]
