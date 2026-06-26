from __future__ import annotations

from pathlib import Path

import typer

from services.dataset.castle_heart_rate import (
    attach_heart_rate_to_events,
    load_heart_rate_samples,
    load_heart_rate_sources,
    load_recording_clock_starts,
    summarize_heart_rate_enrichment,
    write_heart_rate_enrichment_summary,
)
from services.events.manifest import load_event_manifest, write_event_manifest

app = typer.Typer(
    help="Attach CASTLE heart-rate summaries to Event Records using timeline inventory."
)


@app.command()
def main(
    input_manifest: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Input Event Manifest.",
    ),
    output_manifest: Path = typer.Option(
        ...,
        help="Heart-rate-enriched Event Manifest.",
    ),
    timeline_inventory: Path = typer.Option(
        Path("processed/timeline/day1_Allie/source_timeline_inventory.csv"),
        exists=True,
        dir_okay=False,
        help="Timeline source inventory CSV.",
    ),
    day: str = typer.Option("day1"),
    participant_id: str = typer.Option("Allie"),
    min_confidence: float = typer.Option(1.0, min=0.0),
    output_summary: Path | None = typer.Option(
        None,
        help="Optional per-event heart-rate enrichment QA CSV.",
    ),
) -> None:
    events = [
        event
        for event in load_event_manifest(input_manifest)
        if event.participant_id == participant_id
        and event.video_id.startswith(f"{day}_{participant_id}_")
    ]
    recording_starts = load_recording_clock_starts(timeline_inventory)
    heart_sources = load_heart_rate_sources(timeline_inventory)
    source = heart_sources.get((day, participant_id))
    if source is None:
        raise typer.BadParameter(
            f"timeline inventory has no heart-rate source for "
            f"{participant_id}/{day}: {timeline_inventory}"
        )
    samples = load_heart_rate_samples(source.path)
    enriched = attach_heart_rate_to_events(
        events,
        samples,
        recording_clock_starts_ms=recording_starts,
        source_uri=source.source_uri,
        min_confidence=min_confidence,
    )
    write_event_manifest(output_manifest, enriched)
    if output_summary is not None:
        summary_rows = summarize_heart_rate_enrichment(
            events,
            samples,
            recording_clock_starts_ms=recording_starts,
            min_confidence=min_confidence,
        )
        write_heart_rate_enrichment_summary(output_summary, summary_rows)
    attached = sum(event.heart_rate is not None for event in enriched)
    summary_message = f"; wrote QA summary to {output_summary}" if output_summary else ""
    typer.echo(
        f"Wrote {len(enriched)} events to {output_manifest}; "
        f"attached heart-rate summaries to {attached} events"
        f"{summary_message}."
    )


if __name__ == "__main__":
    app()
