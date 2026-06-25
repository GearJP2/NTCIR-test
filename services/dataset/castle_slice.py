from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TranscriptQuality:
    chunk_count: int
    valid_chunk_count: int
    empty_chunk_count: int
    reversed_interval_count: int
    negative_timestamp_count: int
    non_monotonic_count: int
    first_start_sec: float | None
    last_end_sec: float | None


def analyze_transcript(payload: dict[str, Any]) -> TranscriptQuality:
    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("transcript must contain a chunks list")

    valid = 0
    empty = 0
    reversed_intervals = 0
    negative_timestamps = 0
    non_monotonic = 0
    starts: list[float] = []
    ends: list[float] = []
    previous_start: float | None = None

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        timestamp = chunk.get("timestamp")
        if not isinstance(timestamp, list) or len(timestamp) != 2:
            continue
        try:
            start_sec = float(timestamp[0])
            end_sec = float(timestamp[1])
        except (TypeError, ValueError):
            continue

        starts.append(start_sec)
        ends.append(end_sec)
        if not str(chunk.get("text", "")).strip():
            empty += 1
        if start_sec < 0 or end_sec < 0:
            negative_timestamps += 1
        if end_sec < start_sec:
            reversed_intervals += 1
        elif start_sec >= 0:
            valid += 1
        if previous_start is not None and start_sec < previous_start:
            non_monotonic += 1
        previous_start = start_sec

    return TranscriptQuality(
        chunk_count=len(chunks),
        valid_chunk_count=valid,
        empty_chunk_count=empty,
        reversed_interval_count=reversed_intervals,
        negative_timestamp_count=negative_timestamps,
        non_monotonic_count=non_monotonic,
        first_start_sec=min(starts) if starts else None,
        last_end_sec=max(ends) if ends else None,
    )


def transcript_quality_row(
    *,
    day: str,
    participant_id: str,
    recording_stem: str,
    quality: TranscriptQuality,
) -> dict:
    return {
        "day": day,
        "participant_id": participant_id,
        "recording_stem": recording_stem,
        **asdict(quality),
    }


def select_recording_paths(
    paths: list[str],
    *,
    day: str,
    participant_id: str,
) -> dict[str, dict[str, str]]:
    prefix = f"main/{day}/{participant_id}/"
    recordings: dict[str, dict[str, str]] = {}
    for path in paths:
        if not path.startswith(prefix) or path.endswith(".sha256"):
            continue
        relative = path.removeprefix(prefix)
        parts = relative.split("/")
        if len(parts) != 2:
            continue
        modality, filename = parts
        stem, _, extension = filename.partition(".")
        if modality == "video" and extension in {"mp4", "novideo"}:
            recordings.setdefault(stem, {})["video"] = path
        elif modality == "transcript" and extension == "json":
            recordings.setdefault(stem, {})["transcript"] = path
        elif modality == "metadata" and extension:
            recordings.setdefault(stem, {}).setdefault("metadata_prefix", stem)
    return dict(sorted(recordings.items()))
