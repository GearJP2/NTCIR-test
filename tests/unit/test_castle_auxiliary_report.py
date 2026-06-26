import json

from services.dataset.castle_auxiliary_report import (
    build_auxiliary_diagnostics_report,
    write_auxiliary_report_json,
    write_auxiliary_report_markdown,
)


def test_build_auxiliary_diagnostics_report_summarizes_inputs(tmp_path):
    readiness, gaze_streams, gaze_alignment, thermal, violations = write_inputs(tmp_path)

    report = build_auxiliary_diagnostics_report(
        participant_id="Allie",
        day="day1",
        readiness_csv=readiness,
        gaze_streams_csv=gaze_streams,
        gaze_alignment_csv=gaze_alignment,
        thermal_inventory_csv=thermal,
        manifest_violations_csv=violations,
    )

    assert report.attachable_modalities == ["heart_rate"]
    assert report.blocked_modalities == ["gaze", "thermal"]
    assert report.gaze_streams == 1
    assert report.gaze_rows == 10
    assert report.gaze_valid_ratio == 0.6
    assert report.gaze_alignment_candidates == 1
    assert report.gaze_overlapping_candidates == 0
    assert report.thermal_files == 1
    assert report.thermal_unassigned_files == 1
    assert report.thermal_without_timestamp_files == 1
    assert report.manifest_violations == 0


def test_auxiliary_report_writers_emit_markdown_and_json(tmp_path):
    readiness, gaze_streams, gaze_alignment, thermal, violations = write_inputs(tmp_path)
    report = build_auxiliary_diagnostics_report(
        participant_id="Allie",
        day="day1",
        readiness_csv=readiness,
        gaze_streams_csv=gaze_streams,
        gaze_alignment_csv=gaze_alignment,
        thermal_inventory_csv=thermal,
        manifest_violations_csv=violations,
    )
    markdown = tmp_path / "report.md"
    json_path = tmp_path / "report.json"

    write_auxiliary_report_markdown(markdown, report)
    write_auxiliary_report_json(json_path, report)

    assert "Attachable modalities: heart_rate" in markdown.read_text(
        encoding="utf-8"
    )
    assert json.loads(json_path.read_text(encoding="utf-8"))["participant_id"] == "Allie"


def write_inputs(tmp_path):
    readiness = tmp_path / "readiness.csv"
    gaze_streams = tmp_path / "gaze_streams.csv"
    gaze_alignment = tmp_path / "gaze_alignment.csv"
    thermal = tmp_path / "thermal.csv"
    violations = tmp_path / "violations.csv"
    readiness.write_text(
        "participant_id,day,modality,attach_to_event_records,status,evidence_rows,"
        "blocker,next_action\n"
        "Allie,day1,heart_rate,True,attachable_with_clock_day_join,2,,ok\n"
        "Allie,day1,gaze,False,blocked_no_clock_overlap,1,no overlap,map\n"
        "Allie,day1,thermal,False,blocked_unassigned,1,unassigned,map\n",
        encoding="utf-8",
    )
    gaze_streams.write_text(
        "source,media_id,media_name,session_start,first_elapsed_ms,last_elapsed_ms,"
        "duration_ms,row_count,valid_fixation_count,valid_ratio,"
        "unique_fixation_count,aoi_label_count,top_aoi_labels\n"
        "auxiliary/gaze/Allie,0,NewMedia0,start,0,100,100,10,6,0.6,2,0,\n",
        encoding="utf-8",
    )
    gaze_alignment.write_text(
        "gaze_source,media_id,media_name,candidate_anchor,candidate_start_clock_ms,"
        "candidate_end_clock_ms,recording_video_id,recording_start_clock_ms,"
        "recording_end_clock_ms,overlap_ms,overlap_ratio,status,notes\n"
        "auxiliary/gaze/Allie,0,NewMedia0,header,1,2,day1_Allie_08,3,4,0,0,"
        "no_overlap,note\n",
        encoding="utf-8",
    )
    thermal.write_text(
        "source,filename,size_bytes,sequence_index,inferred_day,"
        "inferred_participant_id,timestamp_status,assignment_status,notes\n"
        "auxiliary/thermal/IMG 0003.bmp,IMG 0003.bmp,123,3,,,"
        "no_timestamp_in_path,unassigned,note\n",
        encoding="utf-8",
    )
    violations.write_text(
        "event_id,participant_id,day,video_id,modality,readiness_status,blocker\n",
        encoding="utf-8",
    )
    return readiness, gaze_streams, gaze_alignment, thermal, violations
