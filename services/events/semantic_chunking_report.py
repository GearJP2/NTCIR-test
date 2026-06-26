from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.schemas.event import EventKind
from services.events.manifest import EventManifestError, load_event_manifest


@dataclass(frozen=True)
class SemanticChunkingReport:
    manifest_path: str
    video_ids: list[str]
    selected_transcript_weight: float
    selected_weight_boundary_f1: float | None
    selected_weight_recall_at_1: float | None
    macro_event_count: int
    micro_event_count: int
    transcript_covered_events: int
    heart_rate_covered_events: int
    gaze_covered_events: int
    thermal_covered_events: int
    min_micro_duration_ms: int
    max_micro_duration_ms: int
    mean_micro_duration_ms: float
    manifest_modality_violation_count: int
    chunking_status: str
    blockers: list[str]


def build_semantic_chunking_report(
    *,
    manifest_path: Path,
    transcript_weight: float,
    sweep_summary_path: Path,
    modality_violations_path: Path,
    min_event_ms: int = 10_000,
    max_event_ms: int = 60_000,
) -> SemanticChunkingReport:
    records = load_event_manifest(manifest_path)
    macros = [record for record in records if record.event_kind == EventKind.SEMANTIC_MACRO]
    micros = [record for record in records if record.event_kind == EventKind.SEMANTIC_MICRO]
    blockers: list[str] = []
    if not macros:
        blockers.append("manifest has no semantic_macro events")
    if not micros:
        blockers.append("manifest has no semantic_micro events")

    micro_durations = [record.duration_ms for record in micros]
    too_short = [duration for duration in micro_durations if duration < min_event_ms]
    too_long = [duration for duration in micro_durations if duration > max_event_ms]
    if too_short:
        blockers.append(f"{len(too_short)} micro events shorter than {min_event_ms} ms")
    if too_long:
        blockers.append(f"{len(too_long)} micro events longer than {max_event_ms} ms")

    violations = _read_csv(modality_violations_path)
    if violations:
        blockers.append(f"{len(violations)} modality readiness violations")

    selected_weight = _selected_weight_row(sweep_summary_path, transcript_weight)
    if selected_weight is None:
        blockers.append(
            f"transcript weight {transcript_weight:g} missing from sweep summary"
        )

    return SemanticChunkingReport(
        manifest_path=str(manifest_path),
        video_ids=sorted({record.video_id for record in records}),
        selected_transcript_weight=transcript_weight,
        selected_weight_boundary_f1=(
            float(selected_weight["boundary_f1_micro"])
            if selected_weight is not None
            else None
        ),
        selected_weight_recall_at_1=(
            float(selected_weight["retrieval_recall_at_1"])
            if selected_weight is not None
            else None
        ),
        macro_event_count=len(macros),
        micro_event_count=len(micros),
        transcript_covered_events=sum(record.coverage.transcript for record in records),
        heart_rate_covered_events=sum(record.coverage.heart_rate for record in records),
        gaze_covered_events=sum(record.coverage.gaze for record in records),
        thermal_covered_events=sum(record.coverage.thermal for record in records),
        min_micro_duration_ms=min(micro_durations) if micro_durations else 0,
        max_micro_duration_ms=max(micro_durations) if micro_durations else 0,
        mean_micro_duration_ms=float(np.mean(micro_durations)) if micro_durations else 0.0,
        manifest_modality_violation_count=len(violations),
        chunking_status="ready" if not blockers else "blocked",
        blockers=blockers,
    )


def write_semantic_chunking_report_json(
    path: Path,
    report: SemanticChunkingReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_semantic_chunking_report_markdown(
    path: Path,
    report: SemanticChunkingReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocker_rows = "\n".join(f"- {blocker}" for blocker in report.blockers) or "- none"
    path.write_text(
        f"""# CASTLE Semantic Chunking Report

Manifest: `{report.manifest_path}`

Status: `{report.chunking_status}`

## Configuration

- Transcript boundary weight: `{report.selected_transcript_weight:g}`
- Sweep boundary F1 micro: {_format_optional(report.selected_weight_boundary_f1)}
- Sweep Recall@1: {_format_optional(report.selected_weight_recall_at_1)}

## Manifest summary

- Videos: {", ".join(report.video_ids)}
- Macro events: {report.macro_event_count}
- Micro events: {report.micro_event_count}
- Micro duration range: {report.min_micro_duration_ms}..{report.max_micro_duration_ms} ms
- Mean micro duration: {report.mean_micro_duration_ms:.1f} ms

## Coverage

- Transcript-covered events: {report.transcript_covered_events}
- Heart-rate-covered events: {report.heart_rate_covered_events}
- Gaze-covered events: {report.gaze_covered_events}
- Thermal-covered events: {report.thermal_covered_events}
- Modality readiness violations: {report.manifest_modality_violation_count}

## Blockers

{blocker_rows}
""",
        encoding="utf-8",
    )


def assert_semantic_chunking_ready(report: SemanticChunkingReport) -> None:
    if report.chunking_status != "ready":
        raise EventManifestError(
            "semantic chunking is blocked: " + "; ".join(report.blockers)
        )


def _selected_weight_row(path: Path, transcript_weight: float) -> dict[str, str] | None:
    for row in _read_csv(path):
        if float(row["transcript_weight"]) == transcript_weight:
            return row
    return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _format_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"
