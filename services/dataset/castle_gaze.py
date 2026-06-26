from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from services.dataset.castle_metadata import (
    parse_gaze_session_start,
    parse_seconds_ms,
)


@dataclass(frozen=True)
class GazeStreamSummary:
    source: str
    media_id: str
    media_name: str
    session_start: str
    first_elapsed_ms: int
    last_elapsed_ms: int
    duration_ms: int
    row_count: int
    valid_fixation_count: int
    valid_ratio: float
    unique_fixation_count: int
    aoi_label_count: int
    top_aoi_labels: str


@dataclass(frozen=True)
class RecordingClockWindow:
    video_id: str
    source: str
    start_clock_ms: int
    end_clock_ms: int


@dataclass(frozen=True)
class GazeAlignmentCandidate:
    gaze_source: str
    media_id: str
    media_name: str
    candidate_anchor: str
    candidate_start_clock_ms: int
    candidate_end_clock_ms: int
    recording_video_id: str
    recording_start_clock_ms: int
    recording_end_clock_ms: int
    overlap_ms: int
    overlap_ratio: float
    status: str
    notes: str


def summarize_gaze_streams(path: Path, *, source: str) -> list[GazeStreamSummary]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        fieldnames = list(reader.fieldnames or [])
        time_field = _time_field(fieldnames)
        session_start = parse_gaze_session_start(time_field)
        groups: dict[tuple[str, str], _GazeAccumulator] = {}
        for row in reader:
            media_id = (row.get("MEDIA_ID") or "").strip()
            media_name = (row.get("MEDIA_NAME") or "").strip()
            elapsed_ms = parse_seconds_ms(row[time_field])
            key = (media_id, media_name)
            accumulator = groups.setdefault(
                key,
                _GazeAccumulator(
                    source=source,
                    media_id=media_id,
                    media_name=media_name,
                    session_start=session_start,
                    first_elapsed_ms=elapsed_ms,
                ),
            )
            accumulator.add(row, elapsed_ms)
    return [group.to_summary() for group in groups.values()]


def load_recording_clock_windows(inventory_path: Path) -> list[RecordingClockWindow]:
    windows: list[RecordingClockWindow] = []
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = row["source"]
            if source.startswith("auxiliary/"):
                continue
            start = row.get("first_offset_ms") or ""
            end = row.get("last_offset_ms") or ""
            if not start or not end:
                continue
            windows.append(
                RecordingClockWindow(
                    video_id=_recording_source_to_video_id(source),
                    source=source,
                    start_clock_ms=int(start),
                    end_clock_ms=int(end),
                )
            )
    return windows


def diagnose_gaze_alignment(
    streams: list[GazeStreamSummary],
    recording_windows: list[RecordingClockWindow],
) -> list[GazeAlignmentCandidate]:
    candidates: list[GazeAlignmentCandidate] = []
    for stream in streams:
        for anchor_name, anchor_start_ms, notes in _candidate_anchors(stream):
            start_clock_ms = anchor_start_ms + stream.first_elapsed_ms
            end_clock_ms = anchor_start_ms + stream.last_elapsed_ms
            for window in recording_windows:
                overlap_ms = _overlap_ms(
                    start_clock_ms,
                    end_clock_ms,
                    window.start_clock_ms,
                    window.end_clock_ms,
                )
                duration_ms = max(stream.duration_ms, 1)
                candidates.append(
                    GazeAlignmentCandidate(
                        gaze_source=stream.source,
                        media_id=stream.media_id,
                        media_name=stream.media_name,
                        candidate_anchor=anchor_name,
                        candidate_start_clock_ms=start_clock_ms,
                        candidate_end_clock_ms=end_clock_ms,
                        recording_video_id=window.video_id,
                        recording_start_clock_ms=window.start_clock_ms,
                        recording_end_clock_ms=window.end_clock_ms,
                        overlap_ms=overlap_ms,
                        overlap_ratio=overlap_ms / duration_ms,
                        status="overlaps" if overlap_ms else "no_overlap",
                        notes=notes,
                    )
                )
    return candidates


def write_gaze_stream_summary(
    path: Path,
    rows: list[GazeStreamSummary],
) -> None:
    _write_dataclass_csv(path, rows, GazeStreamSummary)


def write_gaze_alignment_candidates(
    path: Path,
    rows: list[GazeAlignmentCandidate],
) -> None:
    _write_dataclass_csv(path, rows, GazeAlignmentCandidate)


class _GazeAccumulator:
    def __init__(
        self,
        *,
        source: str,
        media_id: str,
        media_name: str,
        session_start: str,
        first_elapsed_ms: int,
    ) -> None:
        self.source = source
        self.media_id = media_id
        self.media_name = media_name
        self.session_start = session_start
        self.first_elapsed_ms = first_elapsed_ms
        self.last_elapsed_ms = first_elapsed_ms
        self.row_count = 0
        self.valid_fixation_count = 0
        self.fixation_ids: set[str] = set()
        self.aoi_counts: dict[str, int] = {}

    def add(self, row: dict[str, str], elapsed_ms: int) -> None:
        self.last_elapsed_ms = elapsed_ms
        self.row_count += 1
        if (row.get("FPOGV") or "").strip() == "1":
            self.valid_fixation_count += 1
            fixation_id = (row.get("FPOGID") or "").strip()
            if fixation_id:
                self.fixation_ids.add(fixation_id)
        aoi = (row.get("AOI") or "").strip()
        if aoi:
            self.aoi_counts[aoi] = self.aoi_counts.get(aoi, 0) + 1

    def to_summary(self) -> GazeStreamSummary:
        duration_ms = max(self.last_elapsed_ms - self.first_elapsed_ms, 0)
        top_aoi = sorted(
            self.aoi_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:5]
        return GazeStreamSummary(
            source=self.source,
            media_id=self.media_id,
            media_name=self.media_name,
            session_start=self.session_start,
            first_elapsed_ms=self.first_elapsed_ms,
            last_elapsed_ms=self.last_elapsed_ms,
            duration_ms=duration_ms,
            row_count=self.row_count,
            valid_fixation_count=self.valid_fixation_count,
            valid_ratio=self.valid_fixation_count / self.row_count
            if self.row_count
            else 0.0,
            unique_fixation_count=len(self.fixation_ids),
            aoi_label_count=sum(self.aoi_counts.values()),
            top_aoi_labels=";".join(
                f"{label}:{count}" for label, count in top_aoi
            ),
        )


def _candidate_anchors(stream: GazeStreamSummary) -> list[tuple[str, int, str]]:
    session_start = datetime.strptime(stream.session_start, "%Y/%m/%d %H:%M:%S.%f")
    header_clock_ms = (
        session_start.hour * 3_600_000
        + session_start.minute * 60_000
        + session_start.second * 1000
        + round(session_start.microsecond / 1000)
    )
    return [
        (
            "header_clock_of_day",
            header_clock_ms,
            "uses gaze header time-of-day; date/timezone unresolved",
        ),
        (
            "elapsed_day_clock",
            0,
            "treats gaze elapsed seconds as day clock; diagnostic only",
        ),
    ]


def _time_field(fieldnames: list[str]) -> str:
    for field in fieldnames:
        if field.startswith("TIME("):
            return field
    raise ValueError("gaze CSV has no TIME(...) field")


def _recording_source_to_video_id(source: str) -> str:
    day, participant_id, filename = source.split("/")
    return f"{day}_{participant_id}_{Path(filename).stem}"


def _overlap_ms(
    left_start_ms: int,
    left_end_ms: int,
    right_start_ms: int,
    right_end_ms: int,
) -> int:
    return max(0, min(left_end_ms, right_end_ms) - max(left_start_ms, right_start_ms))


def _write_dataclass_csv(path: Path, rows: list, row_type: type) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row_type.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
