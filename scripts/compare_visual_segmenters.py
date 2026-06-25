from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import typer

from scripts.build_visual_semantic_events import encode_frames
from services.events.visual_boundaries import VisualSample
from services.events.visual_segmentation import (
    VisualSegmentationConfig,
    run_visual_segmentation,
)

app = typer.Typer(
    help="Compare visual-only CASTLE segmentation detectors on one frame sample."
)


@app.command()
def main(
    frame_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    output_csv: Path = typer.Option(...),
    output_json: Path = typer.Option(...),
    model_name: str = typer.Option("ViT-B-32-quickgelu"),
    pretrained: str = typer.Option("openai"),
    batch_size: int = typer.Option(8, min=1),
    boundary_percentile: float = typer.Option(85.0, min=0.0, max=100.0),
    min_boundary_score: float = typer.Option(0.01, min=0.0),
    min_event_sec: float = typer.Option(10.0, min=1.0),
    max_event_sec: float = typer.Option(60.0, min=1.0),
) -> None:
    paths = sorted(frame_dir.glob("*.jpg"))
    if len(paths) < 2:
        raise typer.BadParameter("at least two sampled frames are required")
    timestamps_ms = [int(path.stem) for path in paths]
    embeddings = encode_frames(
        paths,
        model_name=model_name,
        pretrained=pretrained,
        batch_size=batch_size,
    )
    samples = [
        VisualSample(timestamp_ms=timestamp_ms, embedding=embedding)
        for timestamp_ms, embedding in zip(timestamps_ms, embeddings, strict=True)
    ]
    common = {
        "boundary_percentile": boundary_percentile,
        "min_boundary_score": min_boundary_score,
        "min_event_ms": round(min_event_sec * 1000),
        "max_event_ms": round(max_event_sec * 1000),
    }
    configs = [
        VisualSegmentationConfig(name="v1-adjacent", detector="v1", **common),
        VisualSegmentationConfig(
            name="v2-contextual-r1",
            detector="v2",
            context_radius=1,
            **common,
        ),
        VisualSegmentationConfig(
            name="v2-contextual-r2",
            detector="v2",
            context_radius=2,
            **common,
        ),
        VisualSegmentationConfig(
            name="v2-contextual-r3",
            detector="v2",
            context_radius=3,
            **common,
        ),
    ]
    step_ms = _median_step(timestamps_ms)
    results = [
        run_visual_segmentation(
            samples,
            config,
            start_ms=timestamps_ms[0],
            end_ms=timestamps_ms[-1] + step_ms,
        )
        for config in configs
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        rows = [asdict(result.summary) for result in results]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            [
                {
                    "config": asdict(result.config),
                    "summary": asdict(result.summary),
                    "boundary_timestamps_ms": [
                        boundary.timestamp_ms for boundary in result.boundaries
                    ],
                    "intervals": [asdict(interval) for interval in result.intervals],
                }
                for result in results
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"Wrote {len(results)} detector comparisons to {output_csv}")


def _median_step(timestamps_ms: list[int]) -> int:
    steps = sorted(
        right - left for left, right in zip(timestamps_ms, timestamps_ms[1:])
    )
    return steps[len(steps) // 2]


if __name__ == "__main__":
    app()
