from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.schemas.event import EventRecord


@dataclass(frozen=True)
class TranscriptSpan:
    span_id: str
    video_id: str
    start_ms: int
    end_ms: int
    text: str
    source_index: int


@dataclass(frozen=True)
class RejectedTranscriptSpan:
    video_id: str
    source_index: int
    reason: str
    timestamp: object
    text: str


@dataclass(frozen=True)
class TranscriptCleaningResult:
    spans: list[TranscriptSpan]
    rejected: list[RejectedTranscriptSpan]
    clipped_count: int


def clean_transcript(
    payload: dict[str, Any],
    *,
    video_id: str,
    duration_ms: int,
) -> TranscriptCleaningResult:
    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("transcript must contain a chunks list")

    spans: list[TranscriptSpan] = []
    rejected: list[RejectedTranscriptSpan] = []
    clipped_count = 0

    for source_index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            rejected.append(
                RejectedTranscriptSpan(
                    video_id=video_id,
                    source_index=source_index,
                    reason="chunk_not_object",
                    timestamp=None,
                    text="",
                )
            )
            continue

        text = " ".join(str(chunk.get("text", "")).split())
        timestamp = chunk.get("timestamp")
        reason = _invalid_reason(timestamp, text)
        if reason:
            rejected.append(
                RejectedTranscriptSpan(
                    video_id=video_id,
                    source_index=source_index,
                    reason=reason,
                    timestamp=timestamp,
                    text=text,
                )
            )
            continue

        start_ms = round(float(timestamp[0]) * 1000)
        end_ms = round(float(timestamp[1]) * 1000)
        clipped_start = max(0, min(start_ms, duration_ms))
        clipped_end = max(0, min(end_ms, duration_ms))
        if (clipped_start, clipped_end) != (start_ms, end_ms):
            clipped_count += 1
        if clipped_end <= clipped_start:
            rejected.append(
                RejectedTranscriptSpan(
                    video_id=video_id,
                    source_index=source_index,
                    reason="outside_recording",
                    timestamp=timestamp,
                    text=text,
                )
            )
            continue

        spans.append(
            TranscriptSpan(
                span_id=f"{video_id}:transcript:{source_index:05d}",
                video_id=video_id,
                start_ms=clipped_start,
                end_ms=clipped_end,
                text=text,
                source_index=source_index,
            )
        )

    spans.sort(key=lambda span: (span.start_ms, span.end_ms, span.source_index))
    return TranscriptCleaningResult(
        spans=spans,
        rejected=rejected,
        clipped_count=clipped_count,
    )


def attach_transcripts_to_events(
    events: list[EventRecord],
    spans_by_video: dict[str, list[TranscriptSpan]],
    *,
    source_uri_by_video: dict[str, str] | None = None,
) -> list[EventRecord]:
    source_uri_by_video = source_uri_by_video or {}
    enriched: list[EventRecord] = []

    for event in events:
        overlapping = [
            span
            for span in spans_by_video.get(event.video_id, [])
            if span.start_ms < event.end_ms and span.end_ms > event.start_ms
        ]
        transcript = " ".join(span.text for span in overlapping).strip()
        coverage = event.coverage.model_copy(
            update={"transcript": bool(transcript)}
        )
        raw_evidence_uris = dict(event.raw_evidence_uris)
        source_uri = source_uri_by_video.get(event.video_id)
        if source_uri and transcript:
            raw_evidence_uris["transcript"] = [source_uri]
        enriched.append(
            event.model_copy(
                update={
                    "transcript": transcript,
                    "coverage": coverage,
                    "raw_evidence_uris": raw_evidence_uris,
                }
            )
        )

    return enriched


def _invalid_reason(timestamp: object, text: str) -> str | None:
    if not text:
        return "empty_text"
    if not isinstance(timestamp, list) or len(timestamp) != 2:
        return "invalid_timestamp_shape"
    try:
        start_sec = float(timestamp[0])
        end_sec = float(timestamp[1])
    except (TypeError, ValueError):
        return "non_numeric_timestamp"
    if start_sec < 0 or end_sec < 0:
        return "negative_timestamp"
    if end_sec < start_sec:
        return "reversed_interval"
    if end_sec == start_sec:
        return "zero_duration"
    return None
