from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

import typer

from evaluation.moment_evaluator import run_moment_evaluation
from services.moment_search import MomentSearchService


DEFAULT_PROFILES = [
    "activitynet_visual_only",
    "activitynet_visual_asr_light",
    "activitynet_visual_asr_medium",
    "activitynet_visual_audio_light",
    "activitynet_visual_audio_medium",
    "activitynet_visual_heavy",
]


async def run_profile_sweep(
    manifest_path: Path,
    profiles: list[str],
    output_dir: Path,
    summary_path: Path,
    csv_path: Path,
    write_details: bool = False,
) -> list[dict]:
    service = MomentSearchService()
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for profile in profiles:
        stem = _safe_stem(profile)
        scores = await run_moment_evaluation(
            manifest_path=manifest_path,
            profile_name=profile,
            searcher=service,
            summary_path=output_dir / f"{stem}_summary.json" if write_details else None,
            results_path=output_dir / f"{stem}_results.jsonl" if write_details else None,
            query_csv_path=output_dir / f"{stem}_queries.csv" if write_details else None,
            report_path=output_dir / f"{stem}_report.md" if write_details else None,
        )
        rows.append(scores)
        print(json.dumps(scores, indent=2, sort_keys=True))

    _write_json(summary_path, rows)
    _write_csv(csv_path, rows)
    return rows


def main(
    manifest_path: Path = typer.Option(
        Path("data/manifests/activitynet_dev200.jsonl"),
        help="ActivityNet Evaluation Manifest JSONL path.",
    ),
    profile: list[str] = typer.Option(
        DEFAULT_PROFILES,
        "--profile",
        help="Evaluation Profile to run. Repeat to override the default sweep.",
    ),
    output_dir: Path = typer.Option(
        Path("data/evaluation/profile_sweep"),
        help="Directory for optional per-profile outputs.",
    ),
    summary_path: Path = typer.Option(
        Path("data/evaluation/activitynet_profile_sweep_summary.json"),
        help="JSON path for aggregate sweep metrics.",
    ),
    csv_path: Path = typer.Option(
        Path("data/evaluation/activitynet_profile_sweep_summary.csv"),
        help="CSV path for aggregate sweep metrics.",
    ),
    write_details: bool = typer.Option(
        False,
        help="Write per-profile JSONL/CSV/Markdown reports in output_dir.",
    ),
) -> None:
    asyncio.run(
        run_profile_sweep(
            manifest_path=manifest_path,
            profiles=profile,
            output_dir=output_dir,
            summary_path=summary_path,
            csv_path=csv_path,
            write_details=write_details,
        )
    )


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "profile",
        "num_videos",
        "num_queries",
        "top_k",
        "tiou_threshold",
        "Recall@10",
        "mAP@10",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _safe_stem(profile: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in profile)


if __name__ == "__main__":
    typer.run(main)
