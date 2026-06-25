from app.schemas.event import EventKind, EventRecord, ModalityCoverage
from services.dataset.castle_transcripts import (
    attach_transcripts_to_events,
    clean_transcript,
)


def event(start_ms: int, end_ms: int) -> EventRecord:
    return EventRecord(
        schema_version="1.0",
        processing_version="test",
        event_id=f"V01_F30_{start_ms}",
        participant_id="P01",
        video_id="V01",
        event_kind=EventKind.FIXED_30S,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=end_ms - start_ms,
        video_uri="https://example.com/video.mp4",
        coverage=ModalityCoverage(video=True),
    )


def test_clean_transcript_rejects_invalid_rows_and_sorts_valid_spans():
    result = clean_transcript(
        {
            "chunks": [
                {"timestamp": [5, 6], "text": " second  span "},
                {"timestamp": [3, 2], "text": "reversed"},
                {"timestamp": [1, 2], "text": "first span"},
                {"timestamp": [2, 3], "text": "  "},
            ]
        },
        video_id="V01",
        duration_ms=10_000,
    )

    assert [span.text for span in result.spans] == ["first span", "second span"]
    assert [row.reason for row in result.rejected] == [
        "reversed_interval",
        "empty_text",
    ]


def test_clean_transcript_clips_span_to_recording_duration():
    result = clean_transcript(
        {"chunks": [{"timestamp": [9.5, 10.5], "text": "ending"}]},
        video_id="V01",
        duration_ms=10_000,
    )

    assert result.spans[0].end_ms == 10_000
    assert result.clipped_count == 1


def test_attach_transcripts_uses_temporal_overlap_and_updates_coverage():
    cleaned = clean_transcript(
        {
            "chunks": [
                {"timestamp": [0, 4], "text": "first"},
                {"timestamp": [4, 8], "text": "second"},
            ]
        },
        video_id="V01",
        duration_ms=10_000,
    )

    enriched = attach_transcripts_to_events(
        [event(0, 5_000), event(8_000, 10_000)],
        {"V01": cleaned.spans},
        source_uri_by_video={"V01": "https://example.com/transcript.json"},
    )

    assert enriched[0].transcript == "first second"
    assert enriched[0].coverage.transcript is True
    assert enriched[0].raw_evidence_uris["transcript"]
    assert enriched[1].transcript == ""
    assert enriched[1].coverage.transcript is False
