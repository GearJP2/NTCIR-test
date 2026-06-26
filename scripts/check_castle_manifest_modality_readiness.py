from __future__ import annotations

from pathlib import Path

import typer

from services.dataset.castle_modality_guard import (
    find_blocked_modality_violations,
    load_modality_readiness,
    write_modality_violations,
)
from services.events.manifest import load_event_manifest

app = typer.Typer(
    help="Fail when CASTLE Event Records attach modalities blocked by readiness gates."
)


@app.command()
def main(
    event_manifest: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Event Manifest JSONL to check.",
    ),
    modality_readiness: Path = typer.Option(
        Path("processed/timeline/day1_Allie/modality_readiness.csv"),
        exists=True,
        dir_okay=False,
        help="Modality readiness CSV.",
    ),
    output_violations: Path = typer.Option(
        Path("processed/timeline/day1_Allie/modality_readiness_violations.csv"),
        help="Output violation CSV.",
    ),
) -> None:
    events = load_event_manifest(event_manifest)
    readiness = load_modality_readiness(modality_readiness)
    violations = find_blocked_modality_violations(events, readiness)
    write_modality_violations(output_violations, violations)
    if violations:
        typer.echo(
            f"Found {len(violations)} blocked modality attachments; "
            f"details: {output_violations}"
        )
        raise typer.Exit(1)
    typer.echo(
        f"No blocked modality attachments in {event_manifest}; "
        f"wrote {output_violations}."
    )


if __name__ == "__main__":
    app()
