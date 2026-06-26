from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from services.dataset.castle_audit import CASTLE_REPO_ID
from services.dataset.castle_gaze import (
    diagnose_gaze_alignment,
    load_recording_clock_windows,
    summarize_gaze_streams,
    write_gaze_alignment_candidates,
    write_gaze_stream_summary,
)

app = typer.Typer(
    help="Build CASTLE gaze stream and clock-alignment diagnostics."
)


@app.command()
def main(
    timeline_inventory: Path = typer.Option(
        Path("processed/timeline/day1_Allie/source_timeline_inventory.csv"),
        exists=True,
        dir_okay=False,
        help="Timeline source inventory CSV with recording clock windows.",
    ),
    gaze_csv: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Optional local gaze CSV. Downloads from CASTLE when omitted.",
    ),
    day: str = typer.Option("day1"),
    participant_id: str = typer.Option("Allie"),
    revision: str = typer.Option(
        "c8e7b5cd9e9c83d0ff42560fc1169bed7867abd4",
        help="Pinned CASTLE repository revision.",
    ),
    output_streams: Path = typer.Option(
        Path("processed/timeline/day1_Allie/gaze_stream_summary.csv"),
        help="Output gaze stream summary CSV.",
    ),
    output_alignment: Path = typer.Option(
        Path("processed/timeline/day1_Allie/gaze_alignment_candidates.csv"),
        help="Output candidate alignment CSV.",
    ),
) -> None:
    load_dotenv()
    gaze_path = gaze_csv or Path(
        hf_hub_download(
            CASTLE_REPO_ID,
            f"auxiliary/gaze/{participant_id}.csv",
            repo_type="dataset",
            revision=revision,
            token=True,
        )
    )
    source = f"auxiliary/gaze/{participant_id}"
    streams = summarize_gaze_streams(gaze_path, source=source)
    windows = [
        window
        for window in load_recording_clock_windows(timeline_inventory)
        if window.video_id.startswith(f"{day}_{participant_id}_")
    ]
    candidates = diagnose_gaze_alignment(streams, windows)
    write_gaze_stream_summary(output_streams, streams)
    write_gaze_alignment_candidates(output_alignment, candidates)
    overlapping = sum(candidate.overlap_ms > 0 for candidate in candidates)
    typer.echo(
        f"Wrote {len(streams)} gaze stream rows to {output_streams}; "
        f"{len(candidates)} alignment candidates to {output_alignment}; "
        f"{overlapping} candidates overlap recording clock windows."
    )


if __name__ == "__main__":
    app()
