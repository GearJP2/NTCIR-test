from __future__ import annotations

import csv
import json
from pathlib import Path

import typer


def summarize_temporal_tradeoff(
    summary_paths: list[Path],
    cost_path: Path | None,
    csv_path: Path,
    markdown_path: Path,
    latex_path: Path | None = None,
) -> list[dict]:
    rows = [_normalize_summary(path) for path in summary_paths]
    cost_by_setting = _load_moment_costs(cost_path) if cost_path is not None else {}
    for row in rows:
        cost = cost_by_setting.get(row["temporal_setting"])
        row["candidate_windows"] = cost.get("total_units") if cost else None
        row["window_cost_relative"] = cost.get("relative_to_default") if cost else None

    rows.sort(key=lambda row: (row["window_sec"], row["stride_sec"], row["profile"]))
    _write_csv(csv_path, rows)
    _write_markdown(markdown_path, rows)
    if latex_path is not None:
        _write_latex(latex_path, rows)
    return rows


def main(
    summary_path: list[Path] = typer.Argument(
        ...,
        help="One or more ActivityNet moment evaluator summary JSON files.",
    ),
    cost_path: Path | None = typer.Option(
        Path("data/evaluation/activitynet_ablation_costs.json"),
        help="Optional ablation cost JSON path.",
    ),
    csv_path: Path = typer.Option(
        Path("data/evaluation/activitynet_temporal_tradeoff.csv"),
        help="Output CSV table path.",
    ),
    markdown_path: Path = typer.Option(
        Path("data/evaluation/activitynet_temporal_tradeoff.md"),
        help="Output Markdown table path.",
    ),
    latex_path: Path | None = typer.Option(
        None,
        help="Optional output LaTeX table path.",
    ),
) -> None:
    rows = summarize_temporal_tradeoff(
        summary_paths=summary_path,
        cost_path=cost_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
        latex_path=latex_path,
    )
    print(json.dumps(rows, indent=2, sort_keys=True))


def _normalize_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    top_k = int(data["top_k"])
    recall_key = f"Recall@{top_k}"
    map_key = f"mAP@{top_k}"
    window_sec = float(data.get("window_sec", 10.0))
    stride_sec = float(data.get("stride_sec", 5.0))
    return {
        "profile": data["profile"],
        "temporal_setting": f"{window_sec:g}s/{stride_sec:g}s",
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "num_videos": data["num_videos"],
        "num_queries": data["num_queries"],
        "top_k": top_k,
        "tiou_threshold": data["tiou_threshold"],
        recall_key: data[recall_key],
        map_key: data[map_key],
        "elapsed_sec": data.get("elapsed_sec"),
        "queries_per_sec": data.get("queries_per_sec"),
        "candidate_windows": None,
        "window_cost_relative": None,
    }


def _load_moment_costs(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["setting"]: row
        for row in rows
        if row.get("ablation_type") == "moment_windows"
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_project_rows(rows, fieldnames))


def _write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    lines = [
        "# ActivityNet Temporal Granularity Trade-off",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(field)) for field in fieldnames) + " |")
    lines.extend(
        [
            "",
            "Interpretation: wider temporal windows can improve tIoU-based metrics while returning coarser localized moments. Report window/stride and candidate-window cost with every result.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latex(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    recall_key, map_key = _metric_keys(rows)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{ActivityNet temporal granularity trade-off on the dev200 subset.}",
        "\\label{tab:activitynet-temporal-tradeoff}",
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        "Setting & Windows & Rel. cost & Recall@10 & mAP@10 & Time (s) & Q/s \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["temporal_setting"]),
                    _latex_format_int(row["candidate_windows"]),
                    _latex_format_float(row["window_cost_relative"], precision=3),
                    f"{float(row[recall_key]):.3f}",
                    f"{float(row[map_key]):.3f}",
                    _latex_format_float(row["elapsed_sec"], precision=1),
                    _latex_format_float(row["queries_per_sec"], precision=2),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _fieldnames(rows: list[dict]) -> list[str]:
    metric_keys = sorted(
        {key for row in rows for key in row if key.startswith("Recall@") or key.startswith("mAP@")}
    )
    return [
        "profile",
        "temporal_setting",
        "num_videos",
        "num_queries",
        "top_k",
        "tiou_threshold",
        *metric_keys,
        "candidate_windows",
        "window_cost_relative",
        "elapsed_sec",
        "queries_per_sec",
    ]


def _project_rows(rows: list[dict], fieldnames: list[str]) -> list[dict]:
    return [{field: row.get(field) for field in fieldnames} for row in rows]


def _metric_keys(rows: list[dict]) -> tuple[str, str]:
    recall_keys = sorted({key for row in rows for key in row if key.startswith("Recall@")})
    map_keys = sorted({key for row in rows for key in row if key.startswith("mAP@")})
    if len(recall_keys) != 1 or len(map_keys) != 1:
        raise ValueError("Expected exactly one Recall@K metric and one mAP@K metric")
    return recall_keys[0], map_keys[0]


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else str(value)


def _latex_format_int(value: object) -> str:
    return "--" if value is None else str(value)


def _latex_format_float(value: object, precision: int) -> str:
    return "--" if value is None else f"{float(value):.{precision}f}"


def _latex_escape(value: str) -> str:
    return value.replace("_", "\\_")


if __name__ == "__main__":
    typer.run(main)
