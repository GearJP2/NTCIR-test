from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import typer

from evaluation.manifest import load_evaluation_manifest
from services.retrieval.moments import generate_fixed_windows


def estimate_ablation_costs(
    manifest_path: Path,
    output_csv: Path,
    output_markdown: Path,
    output_json: Path | None = None,
    window_stride: list[str] | None = None,
    keyframe_interval: list[float] | None = None,
) -> list[dict]:
    videos = load_evaluation_manifest(manifest_path)
    window_stride = window_stride or ["10:5", "20:10"]
    keyframe_interval = keyframe_interval or [2.0, 5.0, 10.0]

    rows = []
    for spec in window_stride:
        window_sec, stride_sec = _parse_window_stride(spec)
        total_windows = sum(
            len(
                generate_fixed_windows(
                    video.media_id,
                    duration_sec=video.duration_sec or 0.0,
                    window_sec=window_sec,
                    stride_sec=stride_sec,
                )
            )
            for video in videos
        )
        rows.append(
            {
                "ablation_type": "moment_windows",
                "setting": f"{window_sec:g}s/{stride_sec:g}s",
                "num_videos": len(videos),
                "total_units": total_windows,
                "avg_units_per_video": total_windows / len(videos) if videos else 0.0,
                "relative_to_default": None,
            }
        )

    for interval in keyframe_interval:
        total_keyframes = sum(
            _estimated_keyframes(video.duration_sec or 0.0, interval)
            for video in videos
        )
        rows.append(
            {
                "ablation_type": "visual_keyframes",
                "setting": f"{interval:g}s",
                "num_videos": len(videos),
                "total_units": total_keyframes,
                "avg_units_per_video": total_keyframes / len(videos) if videos else 0.0,
                "relative_to_default": None,
            }
        )

    _add_relative_costs(rows)
    _write_csv(output_csv, rows)
    _write_markdown(output_markdown, rows)
    if output_json is not None:
        _write_json(output_json, rows)
    return rows


def main(
    manifest_path: Path = typer.Option(
        Path("data/manifests/activitynet_dev200.jsonl"),
        help="ActivityNet Evaluation Manifest JSONL path.",
    ),
    output_csv: Path = typer.Option(
        Path("data/evaluation/activitynet_ablation_costs.csv"),
        help="Output CSV path.",
    ),
    output_markdown: Path = typer.Option(
        Path("data/evaluation/activitynet_ablation_costs.md"),
        help="Output Markdown path.",
    ),
    output_json: Path | None = typer.Option(
        None,
        help="Optional output JSON path.",
    ),
    window_stride: list[str] = typer.Option(
        ["10:5", "20:10"],
        "--window-stride",
        help="Moment window/stride setting as WINDOW:STRIDE seconds. Repeatable.",
    ),
    keyframe_interval: list[float] = typer.Option(
        [2.0, 5.0, 10.0],
        "--keyframe-interval",
        help="Visual keyframe interval in seconds. Repeatable.",
    ),
) -> None:
    rows = estimate_ablation_costs(
        manifest_path=manifest_path,
        output_csv=output_csv,
        output_markdown=output_markdown,
        output_json=output_json,
        window_stride=window_stride,
        keyframe_interval=keyframe_interval,
    )
    print(json.dumps(rows, indent=2, sort_keys=True))


def _parse_window_stride(spec: str) -> tuple[float, float]:
    try:
        window_text, stride_text = spec.split(":", maxsplit=1)
        window_sec = float(window_text)
        stride_sec = float(stride_text)
    except ValueError as exc:
        raise ValueError(f"Invalid window stride setting '{spec}', expected WINDOW:STRIDE") from exc
    if window_sec <= 0.0 or stride_sec <= 0.0:
        raise ValueError("window and stride must be positive")
    return window_sec, stride_sec


def _estimated_keyframes(duration_sec: float, interval_sec: float) -> int:
    if interval_sec <= 0.0:
        raise ValueError("keyframe interval must be positive")
    if duration_sec <= 0.0:
        return 0
    return int(math.ceil(duration_sec / interval_sec))


def _add_relative_costs(rows: list[dict]) -> None:
    defaults = {
        "moment_windows": _baseline_units(rows, "moment_windows", "10s/5s"),
        "visual_keyframes": _baseline_units(rows, "visual_keyframes", "2s"),
    }
    for row in rows:
        default = defaults[row["ablation_type"]]
        row["relative_to_default"] = row["total_units"] / default if default else 0.0


def _baseline_units(rows: list[dict], ablation_type: str, preferred_setting: str) -> int:
    typed_rows = [row for row in rows if row["ablation_type"] == ablation_type]
    for row in typed_rows:
        if row["setting"] == preferred_setting:
            return row["total_units"]
    return typed_rows[0]["total_units"] if typed_rows else 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ablation_type",
        "setting",
        "num_videos",
        "total_units",
        "avg_units_per_video",
        "relative_to_default",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ActivityNet Ablation Cost Estimates",
        "",
        "| Type | Setting | Videos | Units | Avg/video | Relative cost |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["ablation_type"],
                    row["setting"],
                    str(row["num_videos"]),
                    str(row["total_units"]),
                    f"{row['avg_units_per_video']:.2f}",
                    f"{row['relative_to_default']:.3f}x",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    typer.run(main)
