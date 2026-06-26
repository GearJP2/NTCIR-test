from __future__ import annotations

from pathlib import Path

import typer

from services.dataset.castle_auxiliary_report import (
    build_auxiliary_diagnostics_report,
    write_auxiliary_report_json,
    write_auxiliary_report_markdown,
)

app = typer.Typer(help="Build a compact CASTLE auxiliary diagnostics report.")


@app.command()
def main(
    participant_id: str = typer.Option("Allie"),
    day: str = typer.Option("day1"),
    readiness_csv: Path = typer.Option(
        Path("processed/timeline/day1_Allie/modality_readiness.csv"),
        exists=True,
        dir_okay=False,
    ),
    gaze_streams_csv: Path = typer.Option(
        Path("processed/timeline/day1_Allie/gaze_stream_summary.csv"),
        exists=True,
        dir_okay=False,
    ),
    gaze_alignment_csv: Path = typer.Option(
        Path("processed/timeline/day1_Allie/gaze_alignment_candidates.csv"),
        exists=True,
        dir_okay=False,
    ),
    thermal_inventory_csv: Path = typer.Option(
        Path("processed/timeline/thermal_inventory.csv"),
        exists=True,
        dir_okay=False,
    ),
    manifest_violations_csv: Path = typer.Option(
        Path("processed/timeline/day1_Allie/modality_readiness_violations.csv"),
        exists=True,
        dir_okay=False,
    ),
    output_markdown: Path = typer.Option(
        Path("processed/timeline/day1_Allie/auxiliary_diagnostics_report.md"),
    ),
    output_json: Path = typer.Option(
        Path("processed/timeline/day1_Allie/auxiliary_diagnostics_report.json"),
    ),
) -> None:
    report = build_auxiliary_diagnostics_report(
        participant_id=participant_id,
        day=day,
        readiness_csv=readiness_csv,
        gaze_streams_csv=gaze_streams_csv,
        gaze_alignment_csv=gaze_alignment_csv,
        thermal_inventory_csv=thermal_inventory_csv,
        manifest_violations_csv=manifest_violations_csv,
    )
    write_auxiliary_report_markdown(output_markdown, report)
    write_auxiliary_report_json(output_json, report)
    typer.echo(
        f"Wrote auxiliary diagnostics report to {output_markdown} and {output_json}."
    )


if __name__ == "__main__":
    app()
