from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from app.schemas.event import EventKind, EventRecord


class EventManifestError(ValueError):
    """Raised when an Event Manifest violates record or cross-record invariants."""


def load_event_manifest(path: Path) -> list[EventRecord]:
    records: list[EventRecord] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            records.append(EventRecord.model_validate_json(line))
        except (ValidationError, ValueError) as exc:
            raise EventManifestError(f"{path}:{line_number}: {exc}") from exc

    validate_event_manifest(records)
    return records


def write_event_manifest(path: Path, records: Iterable[EventRecord]) -> None:
    validated = list(records)
    validate_event_manifest(validated)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
        for record in validated
    )
    path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")


def validate_event_manifest(records: Iterable[EventRecord]) -> None:
    materialized = list(records)
    by_id: dict[str, EventRecord] = {}
    for record in materialized:
        if record.event_id in by_id:
            raise EventManifestError(f"duplicate event_id: {record.event_id}")
        by_id[record.event_id] = record

    for record in materialized:
        if record.event_kind != EventKind.SEMANTIC_MICRO:
            continue
        parent = by_id.get(record.parent_event_id or "")
        if parent is None:
            raise EventManifestError(
                f"{record.event_id}: missing parent event {record.parent_event_id}"
            )
        if parent.event_kind != EventKind.SEMANTIC_MACRO:
            raise EventManifestError(
                f"{record.event_id}: parent must be a semantic_macro event"
            )
        if parent.video_id != record.video_id:
            raise EventManifestError(
                f"{record.event_id}: parent must belong to the same video"
            )
        if parent.start_ms > record.start_ms or parent.end_ms < record.end_ms:
            raise EventManifestError(
                f"{record.event_id}: parent must contain the micro event interval"
            )

    grouped: dict[tuple[str, EventKind], list[EventRecord]] = {}
    for record in materialized:
        if record.event_kind not in {
            EventKind.SEMANTIC_MICRO,
            EventKind.SEMANTIC_MACRO,
        }:
            continue
        grouped.setdefault((record.video_id, record.event_kind), []).append(record)

    for (video_id, event_kind), group in grouped.items():
        ordered = sorted(group, key=lambda record: (record.start_ms, record.end_ms))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_ms < previous.end_ms:
                raise EventManifestError(
                    f"{video_id}: overlapping {event_kind.value} events "
                    f"{previous.event_id} and {current.event_id}"
                )
