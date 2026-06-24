from __future__ import annotations

import csv
import json
from pathlib import Path

import typer


def check_activitynet_paper_artifacts(
    visual_summary_path: Path,
    coarse_summary_path: Path,
    profile_table_csv_path: Path,
    temporal_tradeoff_csv_path: Path,
    cost_json_path: Path,
    findings_path: Path,
    report_draft_path: Path,
    tolerance: float = 1e-6,
) -> list[str]:
    errors: list[str] = []

    visual_summary = _load_json_object(visual_summary_path)
    coarse_summary = _load_json_object(coarse_summary_path)
    costs = _load_json_list(cost_json_path)
    profile_rows = _read_csv(profile_table_csv_path)
    temporal_rows = _read_csv(temporal_tradeoff_csv_path)

    errors.extend(_check_profile_table(profile_rows, visual_summary, tolerance))
    errors.extend(_check_temporal_table(temporal_rows, [visual_summary, coarse_summary], costs, tolerance))
    errors.extend(_check_text_contains(findings_path, _findings_required_text()))
    errors.extend(_check_text_contains(report_draft_path, _draft_required_text()))
    return errors


def main(
    visual_summary_path: Path = typer.Option(
        Path("data/evaluation/activitynet_dev200_visual_only_summary.json"),
        help="Baseline visual-only summary JSON.",
    ),
    coarse_summary_path: Path = typer.Option(
        Path("data/evaluation/activitynet_dev200_visual_only_w20_s10_summary.json"),
        help="Coarse-window visual-only summary JSON.",
    ),
    profile_table_csv_path: Path = typer.Option(
        Path("data/evaluation/activitynet_results_table.csv"),
        help="Profile-ablation CSV table.",
    ),
    temporal_tradeoff_csv_path: Path = typer.Option(
        Path("data/evaluation/activitynet_temporal_tradeoff.csv"),
        help="Temporal trade-off CSV table.",
    ),
    cost_json_path: Path = typer.Option(
        Path("data/evaluation/activitynet_ablation_costs.json"),
        help="Ablation cost JSON.",
    ),
    findings_path: Path = typer.Option(
        Path("data/evaluation/activitynet_findings.md"),
        help="Generated findings Markdown.",
    ),
    report_draft_path: Path = typer.Option(
        Path("docs/REPORT_RESULTS_DRAFT.md"),
        help="Report results draft Markdown.",
    ),
    tolerance: float = typer.Option(1e-6, help="Float comparison tolerance."),
) -> None:
    errors = check_activitynet_paper_artifacts(
        visual_summary_path=visual_summary_path,
        coarse_summary_path=coarse_summary_path,
        profile_table_csv_path=profile_table_csv_path,
        temporal_tradeoff_csv_path=temporal_tradeoff_csv_path,
        cost_json_path=cost_json_path,
        findings_path=findings_path,
        report_draft_path=report_draft_path,
        tolerance=tolerance,
    )
    if errors:
        for error in errors:
            typer.echo(f"ERROR: {error}", err=True)
        raise typer.Exit(code=1)
    typer.echo("ActivityNet paper artifacts are consistent.")


def _check_profile_table(rows: list[dict[str, str]], summary: dict, tolerance: float) -> list[str]:
    errors: list[str] = []
    profile = summary["profile"]
    row = _find_row(rows, "profile", profile)
    if row is None:
        return [f"profile table missing row for {profile}"]

    top_k = int(summary["top_k"])
    recall_key = f"Recall@{top_k}"
    map_key = f"mAP@{top_k}"
    errors.extend(
        [
            *_compare_float(row, recall_key, summary[recall_key], tolerance),
            *_compare_float(row, map_key, summary[map_key], tolerance),
            *_compare_int(row, "num_videos", summary["num_videos"]),
            *_compare_int(row, "num_queries", summary["num_queries"]),
        ]
    )
    return errors


def _check_temporal_table(
    rows: list[dict[str, str]],
    summaries: list[dict],
    costs: list[dict],
    tolerance: float,
) -> list[str]:
    errors: list[str] = []
    cost_by_setting = {
        row["setting"]: row
        for row in costs
        if row.get("ablation_type") == "moment_windows"
    }

    for summary in summaries:
        setting = _temporal_setting(summary)
        row = _find_row(rows, "temporal_setting", setting)
        if row is None:
            errors.append(f"temporal table missing row for {setting}")
            continue

        top_k = int(summary["top_k"])
        recall_key = f"Recall@{top_k}"
        map_key = f"mAP@{top_k}"
        errors.extend(
            [
                *_compare_float(row, recall_key, summary[recall_key], tolerance),
                *_compare_float(row, map_key, summary[map_key], tolerance),
                *_compare_float(row, "elapsed_sec", summary["elapsed_sec"], tolerance),
                *_compare_float(row, "queries_per_sec", summary["queries_per_sec"], tolerance),
            ]
        )

        cost = cost_by_setting.get(setting)
        if cost is None:
            errors.append(f"cost JSON missing moment window setting {setting}")
        else:
            errors.extend(
                [
                    *_compare_int(row, "candidate_windows", cost["total_units"]),
                    *_compare_float(
                        row,
                        "window_cost_relative",
                        cost["relative_to_default"],
                        tolerance,
                    ),
                ]
            )
    return errors


def _check_text_contains(path: Path, required_text: list[str]) -> list[str]:
    if not path.exists():
        return [f"missing text artifact: {path}"]
    text = path.read_text(encoding="utf-8")
    lower_text = text.lower()
    return [
        f"{path} missing required text: {needle}"
        for needle in required_text
        if needle.lower() not in lower_text
    ]


def _findings_required_text() -> list[str]:
    return [
        "controlled proxy benchmark",
        "CASTLE remains the downstream lifelog setting",
        "ActivityNet results should not be described as a direct win",
    ]


def _draft_required_text() -> list[str]:
    return [
        "controlled proxy benchmark",
        "temporal granularity trade-off",
        "Do not claim that this system outperforms WorldMM",
    ]


def _load_json_object(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    return data


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _find_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str] | None:
    for row in rows:
        if row.get(key) == value:
            return row
    return None


def _temporal_setting(summary: dict) -> str:
    return f"{float(summary['window_sec']):g}s/{float(summary['stride_sec']):g}s"


def _compare_float(row: dict[str, str], key: str, expected: float, tolerance: float) -> list[str]:
    if key not in row:
        return [f"CSV row missing field {key}"]
    actual = float(row[key])
    if abs(actual - float(expected)) > tolerance:
        return [f"{key} mismatch: got {actual}, expected {expected}"]
    return []


def _compare_int(row: dict[str, str], key: str, expected: int) -> list[str]:
    if key not in row:
        return [f"CSV row missing field {key}"]
    actual = int(float(row[key]))
    if actual != int(expected):
        return [f"{key} mismatch: got {actual}, expected {expected}"]
    return []


if __name__ == "__main__":
    typer.run(main)
