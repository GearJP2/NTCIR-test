from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

from services.dataset.castle_audit import RepositoryFile


@dataclass(frozen=True)
class ThermalInventoryRow:
    source: str
    filename: str
    size_bytes: int
    sequence_index: int | None
    inferred_day: str
    inferred_participant_id: str
    timestamp_status: str
    assignment_status: str
    notes: str


def inspect_thermal_files(files: list[RepositoryFile]) -> list[ThermalInventoryRow]:
    rows: list[ThermalInventoryRow] = []
    for file in files:
        path = PurePosixPath(file.path)
        parts = path.parts
        if len(parts) < 3 or parts[0:2] != ("auxiliary", "thermal"):
            continue
        if path.suffix.lower() != ".bmp":
            continue
        inferred_day, participant_id = _infer_day_participant(parts)
        rows.append(
            ThermalInventoryRow(
                source=file.path,
                filename=path.name,
                size_bytes=file.size_bytes,
                sequence_index=_sequence_index(path.name),
                inferred_day=inferred_day,
                inferred_participant_id=participant_id,
                timestamp_status=_timestamp_status(path.name),
                assignment_status="unassigned"
                if not inferred_day or not participant_id
                else "path_assigned",
                notes=(
                    "repository path lacks capture timestamp; do not attach to "
                    "Event Records without external manifest or image metadata"
                ),
            )
        )
    return sorted(
        rows,
        key=lambda row: (row.sequence_index is None, row.sequence_index or 0, row.source),
    )


def write_thermal_inventory(path, rows: list[ThermalInventoryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ThermalInventoryRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _sequence_index(filename: str) -> int | None:
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else None


def _timestamp_status(filename: str) -> str:
    if re.search(r"\d{4}[-_/]?\d{2}[-_/]?\d{2}", filename):
        return "date_like_token_in_filename"
    if re.search(r"\d{2}[-_:]?\d{2}[-_:]?\d{2}", filename):
        return "time_like_token_in_filename"
    return "no_timestamp_in_path"


def _infer_day_participant(parts: tuple[str, ...]) -> tuple[str, str]:
    if len(parts) >= 5:
        day = next((part for part in parts if part.startswith("day")), "")
        participant = parts[2] if parts[2] != "thermal" else ""
        return day, participant
    return "", ""
