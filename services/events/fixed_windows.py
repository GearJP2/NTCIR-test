from __future__ import annotations

from app.schemas.event import (
    EventKind,
    EventRecord,
    ModalityCoverage,
    RecordingRecord,
)


def build_fixed_window_events(
    recording: RecordingRecord,
    *,
    event_kind: EventKind,
    processing_version: str,
    schema_version: str = "1.0",
) -> list[EventRecord]:
    window_ms, stride_ms = fixed_window_parameters(event_kind)
    records: list[EventRecord] = []

    start_ms = 0
    index = 1
    while start_ms < recording.duration_ms:
        end_ms = min(start_ms + window_ms, recording.duration_ms)
        records.append(
            EventRecord(
                schema_version=schema_version,
                processing_version=processing_version,
                event_id=_fixed_event_id(recording.video_id, event_kind, index),
                participant_id=recording.participant_id,
                video_id=recording.video_id,
                event_kind=event_kind,
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=end_ms - start_ms,
                video_uri=recording.video_uri,
                coverage=ModalityCoverage(video=True),
            )
        )
        start_ms += stride_ms
        index += 1

    return records


def fixed_window_parameters(event_kind: EventKind) -> tuple[int, int]:
    if event_kind == EventKind.FIXED_30S:
        return 30_000, 20_000
    if event_kind == EventKind.FIXED_120S:
        return 120_000, 120_000
    raise ValueError(f"{event_kind.value} is not a fixed-window event kind")


def _fixed_event_id(video_id: str, event_kind: EventKind, index: int) -> str:
    suffix = "F30" if event_kind == EventKind.FIXED_30S else "F120"
    return f"{video_id}_{suffix}_{index:05d}"
