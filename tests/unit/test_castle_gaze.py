from services.dataset.castle_gaze import (
    diagnose_gaze_alignment,
    load_recording_clock_windows,
    summarize_gaze_streams,
    write_gaze_alignment_candidates,
    write_gaze_stream_summary,
)


def test_summarize_gaze_streams_groups_media_and_valid_fixations(tmp_path):
    path = tmp_path / "gaze.csv"
    path.write_text(
        "MEDIA_ID,MEDIA_NAME,TIME(2024/12/04 16:54:18.272),FPOGV,FPOGID,AOI,\n"
        "0,NewMedia0,0.00000,1,10,screen,\n"
        "0,NewMedia0,0.01000,0,10,,\n"
        "0,NewMedia0,1.00000,1,11,screen,\n",
        encoding="utf-8",
    )

    rows = summarize_gaze_streams(path, source="auxiliary/gaze/Allie")

    assert len(rows) == 1
    assert rows[0].session_start == "2024/12/04 16:54:18.272"
    assert rows[0].duration_ms == 1_000
    assert rows[0].row_count == 3
    assert rows[0].valid_fixation_count == 2
    assert rows[0].unique_fixation_count == 2
    assert rows[0].aoi_label_count == 2
    assert rows[0].top_aoi_labels == "screen:2"


def test_gaze_alignment_candidates_report_no_overlap_for_header_clock(tmp_path):
    gaze = tmp_path / "gaze.csv"
    inventory = tmp_path / "inventory.csv"
    gaze.write_text(
        "MEDIA_ID,MEDIA_NAME,TIME(2024/12/04 16:54:18.272),FPOGV,FPOGID,AOI,\n"
        "0,NewMedia0,0.00000,1,1,,\n"
        "0,NewMedia0,10.00000,1,2,,\n",
        encoding="utf-8",
    )
    inventory.write_text(
        "source,path,time_field,original_time_format,first_time,last_time,"
        "first_offset_ms,last_offset_ms,row_count,anchor_status,notes\n"
        f"day1/Allie/08.ACCL,{tmp_path / '08.ACCL.csv'},time,clock,"
        "08:05:17.180,08:59:59.998,29117180,32399998,10,status,note\n",
        encoding="utf-8",
    )

    streams = summarize_gaze_streams(gaze, source="auxiliary/gaze/Allie")
    candidates = diagnose_gaze_alignment(
        streams,
        load_recording_clock_windows(inventory),
    )

    assert {candidate.candidate_anchor for candidate in candidates} == {
        "header_clock_of_day",
        "elapsed_day_clock",
    }
    assert all(candidate.overlap_ms == 0 for candidate in candidates)
    assert all(candidate.status == "no_overlap" for candidate in candidates)


def test_gaze_diagnostic_writers_emit_headers(tmp_path):
    gaze = tmp_path / "gaze.csv"
    gaze.write_text(
        "MEDIA_ID,MEDIA_NAME,TIME(2024/12/04 16:54:18.272),FPOGV,FPOGID,AOI,\n"
        "0,NewMedia0,0.00000,1,1,,\n",
        encoding="utf-8",
    )
    streams = summarize_gaze_streams(gaze, source="auxiliary/gaze/Allie")
    candidates = diagnose_gaze_alignment(streams, [])
    stream_path = tmp_path / "streams.csv"
    alignment_path = tmp_path / "alignment.csv"

    write_gaze_stream_summary(stream_path, streams)
    write_gaze_alignment_candidates(alignment_path, candidates)

    assert "media_id,media_name" in stream_path.read_text(encoding="utf-8")
    assert "candidate_anchor" in alignment_path.read_text(encoding="utf-8")
