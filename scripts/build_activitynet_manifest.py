from __future__ import annotations

from pathlib import Path

import typer

from evaluation.activitynet_manifest import (
    build_activitynet_manifest_rows,
    write_manifest_jsonl,
)

app = typer.Typer(help="Build an ActivityNet Captions Evaluation Manifest.")


@app.command()
def main(
    output_path: Path = typer.Option(
        Path("data/manifests/activitynet_dev50.jsonl"),
        help="Output JSONL manifest path.",
    ),
    video_root: Path = typer.Option(
        Path("data/activitynet/videos"),
        help="Directory containing resolved ActivityNet video files.",
    ),
    dataset_name: str = typer.Option(
        "friedrichor/ActivityNet_Captions",
        help="Hugging Face dataset name.",
    ),
    split: str = typer.Option("validation", help="Dataset split to sample from."),
    max_videos: int = typer.Option(50, min=1, help="Maximum source videos to include."),
    max_queries: int = typer.Option(500, min=1, help="Maximum Evaluation Queries to include."),
    seed: int = typer.Option(19, help="Fixed sampling seed."),
    allow_missing_video: bool = typer.Option(
        False,
        help="Include rows even when the local video file does not exist.",
    ),
) -> None:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    rows = build_activitynet_manifest_rows(
        dataset_rows=dataset,
        video_root=video_root,
        max_videos=max_videos,
        max_queries=max_queries,
        seed=seed,
        require_video_file=not allow_missing_video,
    )
    write_manifest_jsonl(rows, output_path)

    total_queries = sum(len(row["queries"]) for row in rows)
    typer.echo(
        f"Wrote {len(rows)} videos and {total_queries} Evaluation Queries to {output_path}"
    )


if __name__ == "__main__":
    app()
