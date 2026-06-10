from __future__ import annotations

import random
import os
from pathlib import Path
from typing import Any

import typer

from evaluation.activitynet_manifest import (
    build_activitynet_manifest_rows,
    write_manifest_jsonl,
)

app = typer.Typer(help="Download a small playable ActivityNet validation subset.")


def youtube_id_from_activitynet_video_id(video_id: str) -> str:
    return video_id[2:] if video_id.startswith("v_") else video_id


def video_filename(video_id: str) -> str:
    return f"{video_id}.mp4"


def select_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    return selected


def make_video_paths_relative_to_manifest(
    rows: list[dict[str, Any]], manifest_path: Path
) -> list[dict[str, Any]]:
    manifest_dir = manifest_path.parent
    normalized = []
    for row in rows:
        video_path = Path(str(row["video_path"]))
        if not video_path.is_absolute():
            video_path = video_path.resolve()
        try:
            stored_path = os.path.relpath(video_path, manifest_dir.resolve())
        except ValueError:
            stored_path = str(video_path)
        normalized.append({**row, "video_path": stored_path})
    return normalized


@app.command()
def main(
    output_dir: Path = typer.Option(
        Path("data/activitynet/videos"),
        help="Directory where downloaded videos are stored.",
    ),
    manifest_path: Path = typer.Option(
        Path("data/manifests/activitynet_dev50.jsonl"),
        help="Output Evaluation Manifest JSONL path.",
    ),
    split: str = typer.Option("val1", help="ActivityNet_Captions split to sample."),
    target_videos: int = typer.Option(50, min=1, help="Number of playable videos to keep."),
    max_attempts: int = typer.Option(300, min=1, help="Maximum shuffled rows to try."),
    max_queries: int = typer.Option(500, min=1, help="Maximum Evaluation Queries."),
    seed: int = typer.Option(19, help="Fixed sampling seed."),
    dataset_name: str = typer.Option(
        "friedrichor/ActivityNet_Captions",
        help="Hugging Face dataset name.",
    ),
) -> None:
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name, split=split)
    rows = select_rows([dict(row) for row in dataset], seed=seed)

    kept: list[dict[str, Any]] = []
    attempts = 0
    for row in rows:
        if len(kept) >= target_videos or attempts >= max_attempts:
            break
        attempts += 1

        video_id = str(row["video_id"])
        output_path = output_dir / video_filename(video_id)
        if output_path.exists() and output_path.stat().st_size > 0:
            typer.echo(f"keep existing {video_id}")
            kept.append(row)
            continue

        url = f"https://www.youtube.com/watch?v={youtube_id_from_activitynet_video_id(video_id)}"
        typer.echo(f"download {video_id} ({len(kept) + 1}/{target_videos})")
        if _download_video(url, output_path):
            kept.append(row)
        else:
            typer.echo(f"skip unavailable {video_id}")

    manifest_rows = build_activitynet_manifest_rows(
        dataset_rows=kept,
        video_root=output_dir,
        max_videos=target_videos,
        max_queries=max_queries,
        seed=seed,
        require_video_file=True,
    )
    manifest_rows = make_video_paths_relative_to_manifest(manifest_rows, manifest_path)
    write_manifest_jsonl(manifest_rows, manifest_path)

    total_queries = sum(len(row["queries"]) for row in manifest_rows)
    typer.echo(
        f"Wrote {len(manifest_rows)} videos and {total_queries} Evaluation Queries "
        f"to {manifest_path} after {attempts} attempts"
    )


def _download_video(url: str, output_path: Path) -> bool:
    from yt_dlp import YoutubeDL

    tmp_template = str(output_path.with_suffix(".%(ext)s"))
    options = {
        "format": "b[height<=480][ext=mp4]/b[ext=mp4]/best[height<=480]/best",
        "merge_output_format": "mp4",
        "outtmpl": tmp_template,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 2,
        "fragment_retries": 2,
    }
    try:
        with YoutubeDL(options) as ydl:
            ydl.download([url])
    except Exception:
        return False

    return output_path.exists() and output_path.stat().st_size > 0


if __name__ == "__main__":
    app()
