from __future__ import annotations

import json
from pathlib import Path

import av
import typer

from app.schemas.event import RecordingRecord

app = typer.Typer(help="Sample CASTLE frames through HTTP range requests.")


@app.command()
def main(
    recordings_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    video_id: list[str] = typer.Option(..., help="Repeat for selected video IDs."),
    output_dir: Path = typer.Option(Path("processed/frames")),
    interval_sec: float = typer.Option(600.0, min=1.0),
    start_sec: float = typer.Option(0.0, min=0.0),
    end_sec: float | None = typer.Option(
        None,
        min=0.0,
        help="Optional exclusive end timestamp.",
    ),
) -> None:
    selected = set(video_id)
    recordings = [
        RecordingRecord.model_validate_json(line)
        for line in recordings_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total = 0
    for recording in recordings:
        if recording.video_id not in selected:
            continue
        timestamps = sample_timestamps(
            recording.duration_ms,
            interval_sec,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        total += sample_remote_frames(recording, timestamps, output_dir)
    typer.echo(f"Wrote {total} sampled frames to {output_dir}")


def sample_timestamps(
    duration_ms: int,
    interval_sec: float,
    *,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> list[float]:
    duration_sec = duration_ms / 1000
    stop_sec = min(duration_sec, end_sec) if end_sec is not None else duration_sec
    timestamps: list[float] = []
    timestamp = start_sec
    while timestamp < stop_sec:
        timestamps.append(timestamp)
        timestamp += interval_sec
    return timestamps


def sample_remote_frames(
    recording: RecordingRecord,
    timestamps: list[float],
    output_dir: Path,
) -> int:
    video_dir = output_dir / recording.video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with av.open(recording.video_uri, options={"timeout": "30000000"}) as container:
        stream = next(stream for stream in container.streams if stream.type == "video")
        for timestamp_sec in timestamps:
            seek_offset = int(timestamp_sec / float(stream.time_base))
            container.seek(seek_offset, stream=stream, backward=True)
            frame = next(
                (
                    frame
                    for frame in container.decode(stream)
                    if float(frame.time or 0.0) >= timestamp_sec
                ),
                None,
            )
            if frame is None:
                continue
            image = frame.to_image()
            if image.width > 640:
                height = round(image.height * 640 / image.width)
                image = image.resize((640, height))
            output_path = video_dir / f"{round(timestamp_sec * 1000):010d}.jpg"
            image.save(output_path, format="JPEG", quality=85)
            written += 1
    metadata_path = video_dir / "samples.json"
    metadata_path.write_text(
        json.dumps(
            {
                "video_id": recording.video_id,
                "video_uri": recording.video_uri,
                "timestamps_sec": timestamps,
                "written": written,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return written


if __name__ == "__main__":
    app()
