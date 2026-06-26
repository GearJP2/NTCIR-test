from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuxiliaryDiagnosticsReport:
    participant_id: str
    day: str
    attachable_modalities: list[str]
    blocked_modalities: list[str]
    readiness: list[dict[str, str]]
    gaze_streams: int
    gaze_rows: int
    gaze_valid_ratio: float | None
    gaze_alignment_candidates: int
    gaze_overlapping_candidates: int
    thermal_files: int
    thermal_unassigned_files: int
    thermal_without_timestamp_files: int
    manifest_violations: int


def build_auxiliary_diagnostics_report(
    *,
    participant_id: str,
    day: str,
    readiness_csv: Path,
    gaze_streams_csv: Path,
    gaze_alignment_csv: Path,
    thermal_inventory_csv: Path,
    manifest_violations_csv: Path,
) -> AuxiliaryDiagnosticsReport:
    readiness = _read_csv(readiness_csv)
    gaze_streams = _read_csv(gaze_streams_csv)
    gaze_alignment = _read_csv(gaze_alignment_csv)
    thermal = _read_csv(thermal_inventory_csv)
    violations = _read_csv(manifest_violations_csv)

    return AuxiliaryDiagnosticsReport(
        participant_id=participant_id,
        day=day,
        attachable_modalities=[
            row["modality"]
            for row in readiness
            if _parse_bool(row.get("attach_to_event_records", ""))
        ],
        blocked_modalities=[
            row["modality"]
            for row in readiness
            if not _parse_bool(row.get("attach_to_event_records", ""))
        ],
        readiness=readiness,
        gaze_streams=len(gaze_streams),
        gaze_rows=sum(int(row.get("row_count") or 0) for row in gaze_streams),
        gaze_valid_ratio=_weighted_gaze_valid_ratio(gaze_streams),
        gaze_alignment_candidates=len(gaze_alignment),
        gaze_overlapping_candidates=sum(
            int(float(row.get("overlap_ms") or 0)) > 0 for row in gaze_alignment
        ),
        thermal_files=len(thermal),
        thermal_unassigned_files=sum(
            row.get("assignment_status") == "unassigned" for row in thermal
        ),
        thermal_without_timestamp_files=sum(
            row.get("timestamp_status") == "no_timestamp_in_path" for row in thermal
        ),
        manifest_violations=len(violations),
    )


def write_auxiliary_report_json(path: Path, report: AuxiliaryDiagnosticsReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_auxiliary_report_markdown(
    path: Path,
    report: AuxiliaryDiagnosticsReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readiness_rows = "\n".join(
        "| {modality} | {attach} | {status} | {blocker} |".format(
            modality=row["modality"],
            attach=row["attach_to_event_records"],
            status=row["status"],
            blocker=row["blocker"] or "-",
        )
        for row in report.readiness
    )
    path.write_text(
        f"""# CASTLE Auxiliary Diagnostics Report

Participant/day: `{report.participant_id}/{report.day}`

## Decision

- Attachable modalities: {", ".join(report.attachable_modalities) or "none"}
- Blocked modalities: {", ".join(report.blocked_modalities) or "none"}
- Manifest readiness violations: {report.manifest_violations}

## Readiness gate

| Modality | Attach | Status | Blocker |
|---|---:|---|---|
{readiness_rows}

## Gaze diagnostics

- Streams: {report.gaze_streams}
- Rows: {report.gaze_rows}
- Valid fixation ratio: {_format_ratio(report.gaze_valid_ratio)}
- Alignment candidates: {report.gaze_alignment_candidates}
- Overlapping candidates: {report.gaze_overlapping_candidates}

## Thermal diagnostics

- Thermal files: {report.thermal_files}
- Unassigned files: {report.thermal_unassigned_files}
- Files without path timestamps: {report.thermal_without_timestamp_files}
""",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _weighted_gaze_valid_ratio(rows: list[dict[str, str]]) -> float | None:
    total_rows = sum(int(row.get("row_count") or 0) for row in rows)
    if total_rows == 0:
        return None
    valid_rows = sum(int(row.get("valid_fixation_count") or 0) for row in rows)
    return valid_rows / total_rows


def _format_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"
