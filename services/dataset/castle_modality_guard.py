from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from app.schemas.event import EventRecord
from services.dataset.castle_modality_readiness import ModalityReadinessRow


@dataclass(frozen=True)
class ModalityViolation:
    event_id: str
    participant_id: str
    day: str
    video_id: str
    modality: str
    readiness_status: str
    blocker: str


def load_modality_readiness(path: Path) -> dict[tuple[str, str, str], ModalityReadinessRow]:
    readiness: dict[tuple[str, str, str], ModalityReadinessRow] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = ModalityReadinessRow(
                participant_id=row["participant_id"],
                day=row["day"],
                modality=row["modality"],
                attach_to_event_records=_parse_bool(row["attach_to_event_records"]),
                status=row["status"],
                evidence_rows=int(row["evidence_rows"]),
                blocker=row["blocker"],
                next_action=row["next_action"],
            )
            readiness[(parsed.participant_id, parsed.day, parsed.modality)] = parsed
    return readiness


def find_blocked_modality_violations(
    events: list[EventRecord],
    readiness: dict[tuple[str, str, str], ModalityReadinessRow],
) -> list[ModalityViolation]:
    violations: list[ModalityViolation] = []
    for event in events:
        day = _day_from_video_id(event.video_id)
        for modality in ("heart_rate", "gaze", "thermal"):
            if not getattr(event.coverage, modality):
                continue
            row = readiness.get((event.participant_id, day, modality))
            if row is None:
                violations.append(
                    ModalityViolation(
                        event_id=event.event_id,
                        participant_id=event.participant_id,
                        day=day,
                        video_id=event.video_id,
                        modality=modality,
                        readiness_status="missing_readiness_row",
                        blocker="no readiness row for event participant/day/modality",
                    )
                )
            elif not row.attach_to_event_records:
                violations.append(
                    ModalityViolation(
                        event_id=event.event_id,
                        participant_id=event.participant_id,
                        day=day,
                        video_id=event.video_id,
                        modality=modality,
                        readiness_status=row.status,
                        blocker=row.blocker,
                    )
                )
    return violations


def write_modality_violations(path: Path, violations: list[ModalityViolation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ModalityViolation.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(violation) for violation in violations)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _day_from_video_id(video_id: str) -> str:
    return video_id.split("_", 1)[0]
