from __future__ import annotations

import json
from pathlib import Path

import typer
from pydantic import ValidationError

from app.schemas.event import EventKind, RecordingRecord
from services.events.fixed_windows import build_fixed_window_events
from services.events.manifest import write_event_manifest

app = typer.Typer(
    help="Build fixed-window CASTLE Event Manifests from audited recording JSONL."
)


@app.command()
def main(
    recordings_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="JSONL containing participant_id, video_id, duration_ms, and video_uri.",
    ),
    output_path: Path = typer.Option(
        Path("processed/chunks/fixed_30s.jsonl"),
        help="Output Event Manifest JSONL.",
    ),
    window: str = typer.Option(
        "30s",
        help="Fixed baseline: 30s (10-second overlap) or 120s (non-overlapping).",
    ),
    processing_version: str = typer.Option(
        ...,
        help="Version identifying the dataset preparation run.",
    ),
) -> None:
    event_kind = _event_kind(window)
    events = []
    for line_number, line in enumerate(
        recordings_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            recording = RecordingRecord.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise typer.BadParameter(
                f"{recordings_path}:{line_number}: invalid recording: {exc}"
            ) from exc
        events.extend(
            build_fixed_window_events(
                recording,
                event_kind=event_kind,
                processing_version=processing_version,
            )
        )

    write_event_manifest(output_path, events)
    typer.echo(f"Wrote {len(events)} events to {output_path}")


def _event_kind(window: str) -> EventKind:
    normalized = window.strip().lower()
    if normalized == "30s":
        return EventKind.FIXED_30S
    if normalized == "120s":
        return EventKind.FIXED_120S
    raise typer.BadParameter("window must be either 30s or 120s")


if __name__ == "__main__":
    app()
