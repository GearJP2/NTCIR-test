from __future__ import annotations

import csv
import json
from pathlib import Path

import av
import typer
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

from app.schemas.event import RecordingRecord
from services.dataset.castle_audit import CASTLE_REPO_ID
from services.dataset.castle_slice import (
    analyze_transcript,
    select_recording_paths,
    transcript_quality_row,
)

app = typer.Typer(
    help="Build a small CASTLE recording manifest using remote metadata probing."
)


@app.command()
def main(
    participant_id: str = typer.Option(..., help="CASTLE participant/view directory."),
    day: str = typer.Option(..., help="CASTLE day directory, such as day1."),
    output_dir: Path = typer.Option(
        Path("processed/slices"),
        help="Output root; the participant/day name is appended.",
    ),
    max_recordings: int | None = typer.Option(
        None,
        min=1,
        help="Optional cap for a quick representative slice.",
    ),
) -> None:
    load_dotenv()
    api = HfApi()
    info = api.dataset_info(CASTLE_REPO_ID, token=True)
    revision = info.sha
    recordings = select_recording_paths(
        [file.rfilename for file in info.siblings],
        day=day,
        participant_id=participant_id,
    )
    selected = list(recordings.items())
    if max_recordings is not None:
        selected = selected[:max_recordings]

    slice_dir = output_dir / f"{day}_{participant_id}"
    slice_dir.mkdir(parents=True, exist_ok=True)
    recording_rows: list[RecordingRecord] = []
    quality_rows: list[dict] = []
    source_rows: list[dict] = []

    for stem, sources in selected:
        video_path = sources.get("video")
        if not video_path or video_path.endswith(".novideo"):
            source_rows.append(
                _source_row(day, participant_id, stem, sources, "missing_video")
            )
            continue

        video_uri = hf_hub_url(
            CASTLE_REPO_ID,
            video_path,
            repo_type="dataset",
            revision=revision,
        )
        duration_ms = probe_remote_duration_ms(video_uri)
        recording_rows.append(
            RecordingRecord(
                participant_id=participant_id,
                video_id=f"{day}_{participant_id}_{stem}",
                duration_ms=duration_ms,
                video_uri=video_uri,
            )
        )

        transcript_path = sources.get("transcript")
        transcript_status = "missing"
        if transcript_path:
            local_path = hf_hub_download(
                CASTLE_REPO_ID,
                transcript_path,
                repo_type="dataset",
                revision=revision,
                token=True,
            )
            payload = json.loads(Path(local_path).read_text(encoding="utf-8"))
            quality = analyze_transcript(payload)
            quality_rows.append(
                transcript_quality_row(
                    day=day,
                    participant_id=participant_id,
                    recording_stem=stem,
                    quality=quality,
                )
            )
            transcript_status = (
                "malformed"
                if quality.reversed_interval_count
                or quality.negative_timestamp_count
                or quality.non_monotonic_count
                else "valid"
            )
        source_rows.append(
            _source_row(
                day,
                participant_id,
                stem,
                sources,
                transcript_status,
                duration_ms,
            )
        )

    _write_recordings(slice_dir / "recordings.jsonl", recording_rows)
    _write_csv(slice_dir / "transcript_quality.csv", quality_rows)
    _write_csv(slice_dir / "source_inventory.csv", source_rows)
    (slice_dir / "README.md").write_text(
        _render_report(
            participant_id=participant_id,
            day=day,
            revision=revision,
            recordings=recording_rows,
            quality_rows=quality_rows,
            source_rows=source_rows,
        ),
        encoding="utf-8",
    )
    typer.echo(
        f"Built {len(recording_rows)} recording rows in {slice_dir}; "
        f"validated {len(quality_rows)} transcripts."
    )


def probe_remote_duration_ms(video_uri: str) -> int:
    with av.open(video_uri, options={"timeout": "30000000"}) as container:
        if container.duration is None:
            raise ValueError(f"video has no duration metadata: {video_uri}")
        return round(float(container.duration / av.time_base) * 1000)


def _source_row(
    day: str,
    participant_id: str,
    stem: str,
    sources: dict[str, str],
    status: str,
    duration_ms: int | None = None,
) -> dict:
    return {
        "day": day,
        "participant_id": participant_id,
        "recording_stem": stem,
        "video_path": sources.get("video", ""),
        "transcript_path": sources.get("transcript", ""),
        "metadata_prefix": sources.get("metadata_prefix", ""),
        "duration_ms": duration_ms or "",
        "status": status,
    }


def _write_recordings(path: Path, recordings: list[RecordingRecord]) -> None:
    payload = "\n".join(
        json.dumps(recording.model_dump(mode="json"), ensure_ascii=False)
        for recording in recordings
    )
    path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(
    *,
    participant_id: str,
    day: str,
    revision: str,
    recordings: list[RecordingRecord],
    quality_rows: list[dict],
    source_rows: list[dict],
) -> str:
    malformed = sum(row["status"] == "malformed" for row in source_rows)
    missing_video = sum(row["status"] == "missing_video" for row in source_rows)
    reversed_intervals = sum(
        int(row["reversed_interval_count"]) for row in quality_rows
    )
    total_hours = sum(row.duration_ms for row in recordings) / 3_600_000
    return f"""# CASTLE Representative Slice

- Dataset revision: `{revision}`
- Source: `{day}/{participant_id}`
- Remote-probed recordings: {len(recordings)}
- Total duration: {total_hours:.2f} hours
- Missing video markers: {missing_video}
- Malformed transcript files: {malformed}
- Reversed transcript intervals: {reversed_intervals}

Videos were not downloaded. Durations were read through HTTP range requests.
The generated `recordings.jsonl` is ready for fixed-window Event Manifest
construction.
"""


if __name__ == "__main__":
    app()
