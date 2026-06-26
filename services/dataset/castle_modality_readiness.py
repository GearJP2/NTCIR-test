from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModalityReadinessRow:
    participant_id: str
    day: str
    modality: str
    attach_to_event_records: bool
    status: str
    evidence_rows: int
    blocker: str
    next_action: str


def build_modality_readiness(
    *,
    participant_id: str,
    day: str,
    timeline_inventory: Path,
    gaze_alignment: Path,
    thermal_inventory: Path,
) -> list[ModalityReadinessRow]:
    timeline_rows = _read_csv(timeline_inventory)
    gaze_rows = _read_csv(gaze_alignment)
    thermal_rows = _read_csv(thermal_inventory)
    return [
        _heart_rate_readiness(participant_id, day, timeline_rows),
        _gaze_readiness(participant_id, day, gaze_rows),
        _thermal_readiness(participant_id, day, thermal_rows),
    ]


def write_modality_readiness(
    path: Path,
    rows: list[ModalityReadinessRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ModalityReadinessRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _heart_rate_readiness(
    participant_id: str,
    day: str,
    timeline_rows: list[dict[str, str]],
) -> ModalityReadinessRow:
    source = f"auxiliary/heartrate/{participant_id}/{day}"
    has_source = any(row.get("source") == source for row in timeline_rows)
    has_recording_clocks = any(
        row.get("source", "").startswith(f"{day}/{participant_id}/")
        and row.get("first_offset_ms")
        and row.get("last_offset_ms")
        for row in timeline_rows
    )
    attach = has_source and has_recording_clocks
    return ModalityReadinessRow(
        participant_id=participant_id,
        day=day,
        modality="heart_rate",
        attach_to_event_records=attach,
        status="attachable_with_clock_day_join" if attach else "blocked",
        evidence_rows=int(has_source) + int(has_recording_clocks),
        blocker="" if attach else "missing heart-rate source or recording clock windows",
        next_action=(
            "Keep calendar date/timezone caveat in docs; use QA summary for coverage."
            if attach
            else "Build timeline inventory with heart-rate and recording metadata."
        ),
    )


def _gaze_readiness(
    participant_id: str,
    day: str,
    gaze_rows: list[dict[str, str]],
) -> ModalityReadinessRow:
    scoped = [
        row
        for row in gaze_rows
        if row.get("gaze_source") == f"auxiliary/gaze/{participant_id}"
        and row.get("recording_video_id", "").startswith(f"{day}_{participant_id}_")
    ]
    overlapping = [
        row for row in scoped if int(float(row.get("overlap_ms") or 0)) > 0
    ]
    attach = bool(overlapping)
    return ModalityReadinessRow(
        participant_id=participant_id,
        day=day,
        modality="gaze",
        attach_to_event_records=attach,
        status="candidate_overlap_found" if attach else "blocked_no_clock_overlap",
        evidence_rows=len(scoped),
        blocker="" if attach else "no candidate gaze clock interpretation overlaps recordings",
        next_action=(
            "Inspect overlapping candidate and validate against media/session semantics."
            if attach
            else "Find participant/day gaze session mapping before EventRecord enrichment."
        ),
    )


def _thermal_readiness(
    participant_id: str,
    day: str,
    thermal_rows: list[dict[str, str]],
) -> ModalityReadinessRow:
    scoped = [
        row
        for row in thermal_rows
        if row.get("inferred_participant_id") == participant_id
        and row.get("inferred_day") == day
        and row.get("timestamp_status") != "no_timestamp_in_path"
    ]
    attach = bool(scoped)
    return ModalityReadinessRow(
        participant_id=participant_id,
        day=day,
        modality="thermal",
        attach_to_event_records=attach,
        status="candidate_path_assignment_found" if attach else "blocked_unassigned",
        evidence_rows=len(thermal_rows),
        blocker="" if attach else "thermal files lack participant/day/timestamp assignment",
        next_action=(
            "Validate image timestamps and nearest-event policy."
            if attach
            else "Find external capture manifest or image metadata before enrichment."
        ),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
