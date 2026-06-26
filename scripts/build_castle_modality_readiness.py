from __future__ import annotations

from pathlib import Path

import typer

from services.dataset.castle_modality_readiness import (
    build_modality_readiness,
    write_modality_readiness,
)

app = typer.Typer(
    help="Summarize CASTLE auxiliary modality readiness for EventRecord enrichment."
)


@app.command()
def main(
    participant_id: str = typer.Option("Allie"),
    day: str = typer.Option("day1"),
    timeline_inventory: Path = typer.Option(
        Path("processed/timeline/day1_Allie/source_timeline_inventory.csv"),
        exists=True,
        dir_okay=False,
    ),
    gaze_alignment: Path = typer.Option(
        Path("processed/timeline/day1_Allie/gaze_alignment_candidates.csv"),
        exists=True,
        dir_okay=False,
    ),
    thermal_inventory: Path = typer.Option(
        Path("processed/timeline/thermal_inventory.csv"),
        exists=True,
        dir_okay=False,
    ),
    output_csv: Path = typer.Option(
        Path("processed/timeline/day1_Allie/modality_readiness.csv"),
    ),
) -> None:
    rows = build_modality_readiness(
        participant_id=participant_id,
        day=day,
        timeline_inventory=timeline_inventory,
        gaze_alignment=gaze_alignment,
        thermal_inventory=thermal_inventory,
    )
    write_modality_readiness(output_csv, rows)
    attachable = sum(row.attach_to_event_records for row in rows)
    typer.echo(
        f"Wrote {len(rows)} modality readiness rows to {output_csv}; "
        f"{attachable} modalities attachable."
    )


if __name__ == "__main__":
    app()
