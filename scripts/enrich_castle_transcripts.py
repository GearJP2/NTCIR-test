from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import typer
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, hf_hub_url

from services.dataset.castle_audit import CASTLE_REPO_ID
from services.dataset.castle_transcripts import (
    TranscriptSpan,
    attach_transcripts_to_events,
    clean_transcript,
)
from services.events.manifest import load_event_manifest, write_event_manifest

app = typer.Typer(
    help="Clean CASTLE transcripts and attach overlapping spans to Event Records."
)


@app.command()
def main(
    input_manifest: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Fixed-window or semantic Event Manifest.",
    ),
    output_manifest: Path = typer.Option(
        ...,
        help="Transcript-enriched Event Manifest.",
    ),
    day: str = typer.Option(...),
    participant_id: str = typer.Option(...),
    recording_stem: list[str] = typer.Option(
        ...,
        help="Repeat for each recording stem, for example 08, 09, and 10.",
    ),
    revision: str = typer.Option(..., help="Pinned CASTLE repository revision."),
    cleaned_spans_path: Path = typer.Option(...),
    rejected_spans_path: Path = typer.Option(...),
) -> None:
    load_dotenv()
    selected_video_ids = {
        f"{day}_{participant_id}_{stem}" for stem in recording_stem
    }
    events = [
        event
        for event in load_event_manifest(input_manifest)
        if event.video_id in selected_video_ids
    ]
    duration_by_video = {
        event.video_id: max(
            candidate.end_ms
            for candidate in events
            if candidate.video_id == event.video_id
        )
        for event in events
    }

    spans_by_video: dict[str, list[TranscriptSpan]] = {}
    source_uri_by_video: dict[str, str] = {}
    rejected_rows: list[dict] = []
    clipped_total = 0

    for stem in recording_stem:
        video_id = f"{day}_{participant_id}_{stem}"
        transcript_repo_path = (
            f"main/{day}/{participant_id}/transcript/{stem}.json"
        )
        local_path = hf_hub_download(
            CASTLE_REPO_ID,
            transcript_repo_path,
            repo_type="dataset",
            revision=revision,
            token=True,
        )
        payload = json.loads(Path(local_path).read_text(encoding="utf-8"))
        result = clean_transcript(
            payload,
            video_id=video_id,
            duration_ms=duration_by_video[video_id],
        )
        spans_by_video[video_id] = result.spans
        rejected_rows.extend(asdict(row) for row in result.rejected)
        clipped_total += result.clipped_count
        source_uri_by_video[video_id] = hf_hub_url(
            CASTLE_REPO_ID,
            transcript_repo_path,
            repo_type="dataset",
            revision=revision,
        )

    enriched = attach_transcripts_to_events(
        events,
        spans_by_video,
        source_uri_by_video=source_uri_by_video,
    )
    write_event_manifest(output_manifest, enriched)
    _write_spans(cleaned_spans_path, spans_by_video)
    _write_rejections(rejected_spans_path, rejected_rows)
    typer.echo(
        f"Wrote {len(enriched)} enriched events; "
        f"{sum(len(spans) for spans in spans_by_video.values())} cleaned spans; "
        f"{len(rejected_rows)} rejected; {clipped_total} clipped."
    )


def _write_spans(
    path: Path,
    spans_by_video: dict[str, list[TranscriptSpan]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        asdict(span)
        for video_spans in spans_by_video.values()
        for span in video_spans
    ]
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")


def _write_rejections(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video_id", "source_index", "reason", "timestamp", "text"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "timestamp": json.dumps(row["timestamp"]),
                }
            )


if __name__ == "__main__":
    app()
