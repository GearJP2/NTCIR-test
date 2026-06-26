from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimelineSourceInspection:
    source: str
    path: str
    time_field: str
    original_time_format: str
    first_time: str
    last_time: str
    first_offset_ms: int | None
    last_offset_ms: int | None
    row_count: int
    anchor_status: str
    notes: str = ""


def inspect_main_metadata_csv(
    path: Path,
    *,
    source: str,
) -> TimelineSourceInspection:
    row_count, header, first, last = _scan_csv(path)
    time_field = header[0] if header else "time"
    first_time = first.get(time_field, "") if first else ""
    last_time = last.get(time_field, "") if last else ""
    return TimelineSourceInspection(
        source=source,
        path=str(path),
        time_field=time_field,
        original_time_format="clock-of-day HH:MM:SS.sss",
        first_time=first_time,
        last_time=last_time,
        first_offset_ms=parse_clock_time_ms(first_time) if first_time else None,
        last_offset_ms=parse_clock_time_ms(last_time) if last_time else None,
        row_count=row_count,
        anchor_status="date/timezone unresolved",
        notes="clock-of-day must be combined with CASTLE day date and timezone",
    )


def inspect_heart_rate_csv(path: Path, *, source: str) -> TimelineSourceInspection:
    row_count, header, first, last = _scan_csv(path)
    time_field = header[0] if header else "time"
    first_time = first.get(time_field, "") if first else ""
    last_time = last.get(time_field, "") if last else ""
    return TimelineSourceInspection(
        source=source,
        path=str(path),
        time_field=time_field,
        original_time_format="elapsed HH:MM:SS.s",
        first_time=first_time,
        last_time=last_time,
        first_offset_ms=parse_elapsed_time_ms(first_time) if first_time else None,
        last_offset_ms=parse_elapsed_time_ms(last_time) if last_time else None,
        row_count=row_count,
        anchor_status="participant/day session start unresolved",
        notes="elapsed samples require an external session anchor",
    )


def inspect_gaze_csv(path: Path, *, source: str) -> TimelineSourceInspection:
    row_count, header, first, last = _scan_csv(path)
    time_field = next((field for field in header if field.startswith("TIME(")), "")
    first_time = first.get(time_field, "") if first and time_field else ""
    last_time = last.get(time_field, "") if last and time_field else ""
    return TimelineSourceInspection(
        source=source,
        path=str(path),
        time_field=time_field,
        original_time_format="header session start plus elapsed TIME seconds",
        first_time=first_time,
        last_time=last_time,
        first_offset_ms=parse_seconds_ms(first_time) if first_time else None,
        last_offset_ms=parse_seconds_ms(last_time) if last_time else None,
        row_count=row_count,
        anchor_status="header session start present; timezone unresolved",
        notes=f"session_start={parse_gaze_session_start(time_field)}",
    )


def parse_clock_time_ms(value: str) -> int:
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?\s*", value)
    if not match:
        raise ValueError(f"invalid clock time: {value!r}")
    hours, minutes, seconds, fraction = match.groups()
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1000
        + _fraction_to_ms(fraction)
    )


def parse_elapsed_time_ms(value: str) -> int:
    return parse_clock_time_ms(value)


def parse_seconds_ms(value: str) -> int:
    return round(float(value) * 1000)


def parse_gaze_session_start(header_field: str) -> str:
    match = re.fullmatch(r"TIME\(([^)]+)\)", header_field)
    if not match:
        raise ValueError(f"invalid gaze TIME header: {header_field!r}")
    return match.group(1)


def write_timeline_inventory(
    path: Path,
    inspections: list[TimelineSourceInspection],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(inspection) for inspection in inspections]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scan_csv(path: Path) -> tuple[int, list[str], dict[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        header = list(reader.fieldnames or [])
        first: dict[str, str] = {}
        last: dict[str, str] = {}
        row_count = 0
        for row in reader:
            cleaned = {key.strip(): value.strip() for key, value in row.items() if key}
            if row_count == 0:
                first = cleaned
            last = cleaned
            row_count += 1
    return row_count, header, first, last


def _fraction_to_ms(fraction: str | None) -> int:
    if not fraction:
        return 0
    return round(float(f"0.{fraction}") * 1000)
