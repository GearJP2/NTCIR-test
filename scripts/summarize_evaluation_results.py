from __future__ import annotations

import csv
import json
from pathlib import Path

import typer


def summarize_results(summary_paths: list[Path], csv_path: Path, markdown_path: Path) -> list[dict]:
    rows_by_profile = {
        row["profile"]: row
        for path in summary_paths
        for row in _load_summary_rows(path)
    }
    rows = list(rows_by_profile.values())
    rows.sort(key=lambda row: row["profile"])
    _write_csv(csv_path, rows)
    _write_markdown(markdown_path, rows)
    return rows


def main(
    summary_path: list[Path] = typer.Argument(
        ...,
        help="One or more evaluator summary JSON files.",
    ),
    csv_path: Path = typer.Option(
        Path("data/evaluation/activitynet_results_table.csv"),
        help="Output CSV table path.",
    ),
    markdown_path: Path = typer.Option(
        Path("data/evaluation/activitynet_results_table.md"),
        help="Output Markdown table path.",
    ),
) -> None:
    rows = summarize_results(summary_path, csv_path=csv_path, markdown_path=markdown_path)
    print(json.dumps(rows, indent=2, sort_keys=True))


def _load_summary_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else [data]
    return [_normalize_summary(item) for item in items]


def _normalize_summary(data: dict) -> dict:
    top_k = int(data["top_k"])
    recall_key = f"Recall@{top_k}"
    map_key = f"mAP@{top_k}"
    return {
        "profile": data["profile"],
        "num_videos": data["num_videos"],
        "num_queries": data["num_queries"],
        "top_k": top_k,
        "tiou_threshold": data["tiou_threshold"],
        recall_key: data[recall_key],
        map_key: data[map_key],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    lines = [
        "# ActivityNet Evaluation Results",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(field)) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fieldnames(rows: list[dict]) -> list[str]:
    metric_keys = sorted(
        {key for row in rows for key in row if key.startswith("Recall@") or key.startswith("mAP@")}
    )
    return ["profile", "num_videos", "num_queries", "top_k", "tiou_threshold", *metric_keys]


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else str(value)


if __name__ == "__main__":
    typer.run(main)
