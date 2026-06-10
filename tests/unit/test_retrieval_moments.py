from services.retrieval.moments import (
    EvidenceHit,
    evidence_hits_to_video_moments,
    generate_fixed_windows,
)


def test_generate_fixed_windows_for_exact_duration():
    windows = generate_fixed_windows("v_123", duration_sec=120.0)

    assert len(windows) == 23
    assert windows[0].start_sec == 0.0
    assert windows[0].end_sec == 10.0
    assert windows[-1].start_sec == 110.0
    assert windows[-1].end_sec == 120.0


def test_generate_fixed_windows_adds_tail_window():
    windows = generate_fixed_windows("v_123", duration_sec=123.0)

    assert windows[-2].start_sec == 110.0
    assert windows[-2].end_sec == 120.0
    assert windows[-1].start_sec == 113.0
    assert windows[-1].end_sec == 123.0


def test_generate_fixed_windows_for_short_video():
    windows = generate_fixed_windows("v_123", duration_sec=4.0)

    assert len(windows) == 1
    assert windows[0].start_sec == 0.0
    assert windows[0].end_sec == 4.0


def test_evidence_hits_to_video_moments_maps_timestamp_hit_to_window():
    windows = generate_fixed_windows("v_123", duration_sec=30.0)
    hits = [
        EvidenceHit(
            source_type="visual",
            media_id="v_123",
            score=0.8,
            source_id="frame-1",
            timestamp_sec=12.0,
        )
    ]

    moments = evidence_hits_to_video_moments("v_123", windows, hits)

    assert len(moments) == 2
    assert moments[0].moment_id == "v_123:5.000-15.000"
    assert moments[0].evidence[0].source_type == "visual"
    assert moments[0].evidence[0].timestamp_sec == 12.0


def test_evidence_hits_to_video_moments_maps_interval_hit_to_overlapping_windows():
    windows = generate_fixed_windows("v_123", duration_sec=30.0)
    hits = [
        EvidenceHit(
            source_type="asr",
            media_id="v_123",
            score=0.7,
            source_id="asr-1",
            start_sec=8.0,
            end_sec=13.0,
            text="doing sit ups",
        )
    ]

    moments = evidence_hits_to_video_moments("v_123", windows, hits)

    assert [moment.moment_id for moment in moments] == [
        "v_123:0.000-10.000",
        "v_123:5.000-15.000",
        "v_123:10.000-20.000",
    ]


def test_evidence_hits_to_video_moments_applies_source_weights():
    windows = generate_fixed_windows("v_123", duration_sec=30.0)
    hits = [
        EvidenceHit(
            source_type="asr",
            media_id="v_123",
            score=0.9,
            start_sec=0.0,
            end_sec=10.0,
        ),
        EvidenceHit(
            source_type="visual",
            media_id="v_123",
            score=0.6,
            timestamp_sec=17.0,
        ),
    ]

    moments = evidence_hits_to_video_moments(
        "v_123",
        windows,
        hits,
        source_weights={"visual": 2.0, "audio": 1.0, "asr": 0.5, "summary": 1.0},
    )

    assert moments[0].moment_id == "v_123:10.000-20.000"
    assert moments[0].score == 1.2


def test_evidence_hits_to_video_moments_clamps_negative_scores_for_response_schema():
    windows = generate_fixed_windows("v_123", duration_sec=30.0)
    hits = [
        EvidenceHit(
            source_type="visual",
            media_id="v_123",
            score=-0.04,
            source_id="frame-1",
            timestamp_sec=12.0,
        )
    ]

    moments = evidence_hits_to_video_moments("v_123", windows, hits)

    assert len(moments) == 2
    assert moments[0].score == 0.0
    assert moments[0].evidence[0].score == 0.0


def test_evidence_hits_to_video_moments_ignores_other_media():
    windows = generate_fixed_windows("v_123", duration_sec=30.0)
    hits = [
        EvidenceHit(
            source_type="visual",
            media_id="v_other",
            score=0.9,
            timestamp_sec=12.0,
        )
    ]

    assert evidence_hits_to_video_moments("v_123", windows, hits) == []
