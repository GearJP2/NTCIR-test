from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import av
import typer
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_url

from app.schemas.event import RecordingRecord
from services.dataset.castle_audit import CASTLE_REPO_ID

app = typer.Typer(
    help="Build a CASTLE RecordingRecord manifest for all main video recordings."
)


@app.command()
def main(
    output_path: Path = typer.Option(
        Path("processed/all_castle/recordings.jsonl"),
        help="Output JSONL of RecordingRecord rows.",
    ),
    failures_path: Path = typer.Option(
        Path("processed/all_castle/recording_failures.csv"),
        help="Output CSV of recordings whose duration could not be probed.",
    ),
    max_recordings: int | None = typer.Option(
        None,
        min=1,
        help="Optional cap for smoke tests.",
    ),
    workers: int = typer.Option(8, min=1, help="Concurrent duration probes."),
) -> None:
    load_dotenv()
    api = HfApi()
    info = api.dataset_info(CASTLE_REPO_ID, token=True)
    revision = info.sha
    video_paths = sorted(
        sibling.rfilename
        for sibling in info.siblings
        if _is_main_video(sibling.rfilename)
    )
    if max_recordings is not None:
        video_paths = video_paths[:max_recordings]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_by_video_id = {
        row.video_id: row for row in _load_existing_recordings(output_path)
    }
    failures: list[dict] = []
    pending = [
        path
        for path in video_paths
        if _video_id_for_path(path) not in rows_by_video_id
    ]
    typer.echo(
        f"Found {len(rows_by_video_id)} existing recordings; "
        f"probing {len(pending)} with {workers} workers."
    )

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_probe_video_path, video_path, revision): video_path
            for video_path in pending
        }
        for future in as_completed(futures):
            completed += 1
            video_path = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - recorded for handoff audit.
                video_id = _video_id_for_path(video_path)
                failures.append(
                    {
                        "video_path": video_path,
                        "video_id": video_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                typer.echo(
                    f"[{completed}/{len(pending)}] failed {video_id}: {exc}"
                )
                continue
            rows_by_video_id[row.video_id] = row
            if completed % 25 == 0 or completed == len(pending):
                _write_jsonl(output_path, _sorted_recordings(rows_by_video_id))
                typer.echo(
                    f"[{completed}/{len(pending)}] checkpoint: "
                    f"{len(rows_by_video_id)} recordings"
                )

    rows = _sorted_recordings(rows_by_video_id)
    _write_jsonl(output_path, rows)
    _write_csv(failures_path, failures)
    typer.echo(
        f"Wrote {len(rows)} recordings to {output_path}; "
        f"{len(failures)} failures to {failures_path}."
    )


def probe_remote_duration_ms(video_uri: str) -> int:
    with av.open(video_uri, options={"timeout": "30000000"}) as container:
        if container.duration is None:
            raise ValueError(f"video has no duration metadata: {video_uri}")
        return round(float(container.duration / av.time_base) * 1000)


def _probe_video_path(video_path: str, revision: str) -> RecordingRecord:
    day, participant_id, stem = _parse_main_video_path(video_path)
    video_uri = hf_hub_url(
        CASTLE_REPO_ID,
        video_path,
        repo_type="dataset",
        revision=revision,
    )
    return RecordingRecord(
        participant_id=participant_id,
        video_id=f"{day}_{participant_id}_{stem}",
        duration_ms=probe_remote_duration_ms(video_uri),
        video_uri=video_uri,
    )


def _is_main_video(path: str) -> bool:
    parts = path.split("/")
    return (
        len(parts) == 5
        and parts[0] == "main"
        and parts[3] == "video"
        and path.endswith(".mp4")
    )


def _parse_main_video_path(path: str) -> tuple[str, str, str]:
    parts = path.split("/")
    if not _is_main_video(path):
        raise ValueError(f"not a main video path: {path}")
    stem = Path(parts[4]).stem
    return parts[1], parts[2], stem


def _video_id_for_path(path: str) -> str:
    day, participant_id, stem = _parse_main_video_path(path)
    return f"{day}_{participant_id}_{stem}"


def _load_existing_recordings(path: Path) -> list[RecordingRecord]:
    if not path.exists():
        return []
    return [
        RecordingRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sorted_recordings(rows_by_video_id: dict[str, RecordingRecord]) -> list[RecordingRecord]:
    return [rows_by_video_id[key] for key in sorted(rows_by_video_id)]


def _write_jsonl(path: Path, rows: list[RecordingRecord]) -> None:
    payload = "\n".join(
        json.dumps(row.model_dump(mode="json"), ensure_ascii=False)
        for row in rows
    )
    path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("video_path,video_id,error\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    app()
