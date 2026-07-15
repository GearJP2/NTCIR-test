from __future__ import annotations

from pathlib import Path

import typer

from services.events.semantic_chunking_report import (
    assert_semantic_chunking_ready,
    build_semantic_chunking_report,
    write_semantic_chunking_report_json,
    write_semantic_chunking_report_markdown,
)

app = typer.Typer(
    help="Validate and report the finalized CASTLE semantic chunking artifact."
)


@app.command()
def main(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False),
    transcript_weight: float = typer.Option(0.25, min=0.0, max=1.0),
    sweep_summary: Path = typer.Option(
        Path("processed/semantic/transcript_weight_sweep_summary.csv"),
        exists=True,
        dir_okay=False,
    ),
    modality_violations: Path = typer.Option(
        Path("processed/timeline/day1_Allie/modality_readiness_violations.csv"),
        exists=True,
        dir_okay=False,
    ),
    min_event_ms: int = typer.Option(10_000, min=1),
    max_event_ms: int = typer.Option(60_000, min=1),
    output_markdown: Path = typer.Option(
        Path("processed/semantic/final_semantic_chunking_report.md"),
    ),
    output_json: Path = typer.Option(
        Path("processed/semantic/final_semantic_chunking_report.json"),
    ),
    fail_on_blocked: bool = typer.Option(True),
) -> None:
    report = build_semantic_chunking_report(
        manifest_path=manifest,
        transcript_weight=transcript_weight,
        sweep_summary_path=sweep_summary,
        modality_violations_path=modality_violations,
        min_event_ms=min_event_ms,
        max_event_ms=max_event_ms,
    )
    write_semantic_chunking_report_markdown(output_markdown, report)
    write_semantic_chunking_report_json(output_json, report)
    if fail_on_blocked:
        assert_semantic_chunking_ready(report)
    typer.echo(
        f"Semantic chunking status: {report.chunking_status}; "
        f"wrote {output_markdown} and {output_json}."
    )


if __name__ == "__main__":
    app()
