from __future__ import annotations

import csv
import json
from pathlib import Path

import typer


def summarize_results(
    summary_paths: list[Path],
    csv_path: Path,
    markdown_path: Path,
    latex_path: Path | None = None,
    findings_path: Path | None = None,
    baseline_profile: str = "activitynet_visual_only",
) -> list[dict]:
    rows_by_profile = {
        row["profile"]: row
        for path in summary_paths
        for row in _load_summary_rows(path)
    }
    rows = list(rows_by_profile.values())
    rows.sort(key=lambda row: row["profile"])
    _write_csv(csv_path, rows)
    _write_markdown(markdown_path, rows)
    if latex_path is not None:
        _write_latex(latex_path, rows)
    if findings_path is not None:
        _write_findings(findings_path, rows, baseline_profile=baseline_profile)
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
    latex_path: Path | None = typer.Option(
        None,
        help="Optional output LaTeX table path.",
    ),
    findings_path: Path | None = typer.Option(
        None,
        help="Optional output Markdown findings summary path.",
    ),
    baseline_profile: str = typer.Option(
        "activitynet_visual_only",
        help="Profile to treat as the baseline in the findings summary.",
    ),
) -> None:
    rows = summarize_results(
        summary_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
        latex_path=latex_path,
        findings_path=findings_path,
        baseline_profile=baseline_profile,
    )
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


def _write_latex(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    recall_key, map_key = _metric_keys(rows)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{ActivityNet Captions moment retrieval results on the dev200 subset.}",
        "\\label{tab:activitynet-dev200-results}",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Profile & Videos & Queries & Recall@10 & mAP@10 \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(row["profile"])),
                    str(row["num_videos"]),
                    str(row["num_queries"]),
                    f"{float(row[recall_key]):.3f}",
                    f"{float(row[map_key]):.3f}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_findings(path: Path, rows: list[dict], baseline_profile: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    recall_key, map_key = _metric_keys(rows)
    baseline = _find_profile(rows, baseline_profile)
    best_map_value = max(row[map_key] for row in rows)
    best_recall_value = max(row[recall_key] for row in rows)
    best_map_profiles = [row["profile"] for row in rows if row[map_key] == best_map_value]
    best_recall_profiles = [row["profile"] for row in rows if row[recall_key] == best_recall_value]
    tied_with_baseline = [
        row["profile"]
        for row in rows
        if row["profile"] != baseline_profile
        and row[recall_key] == baseline[recall_key]
        and row[map_key] == baseline[map_key]
    ]
    lower_than_baseline = [
        row["profile"]
        for row in rows
        if row["profile"] != baseline_profile and row[map_key] < baseline[map_key]
    ]
    lines = [
        "# ActivityNet Experiment Findings",
        "",
        "## Protocol",
        "",
        f"- Dataset: ActivityNet Captions dev200 subset ({baseline['num_videos']} videos, {baseline['num_queries']} sentence-level queries).",
        "- Role: controlled proxy benchmark for the temporal grounding component.",
        "- Search scope: single-video moment retrieval.",
        f"- Metrics: Recall@10 and mAP@10 with tIoU >= {baseline['tiou_threshold']}.",
        f"- Current baseline profile: `{baseline_profile}`.",
        "- CASTLE remains the downstream lifelog setting, but is not used for tIoU-based quantitative claims because it lacks timestamped Ground Truth Moments.",
        "",
        "## Key Result",
        "",
        (
            f"`{baseline_profile}` reaches Recall@10 = {_format_metric(baseline[recall_key])} "
            f"and mAP@10 = {_format_metric(baseline[map_key])}."
        ),
        (
            "The best mAP value is "
            f"{_format_metric(best_map_value)}, reached by {_format_profile_list(best_map_profiles)}; "
            "the best Recall@10 value is "
            f"{_format_metric(best_recall_value)}, reached by {_format_profile_list(best_recall_profiles)}."
        ),
        "",
        "## Interpretation",
        "",
    ]
    if tied_with_baseline:
        lines.append(
            "Light ASR/audio fusion ties the visual-only baseline for both Recall@10 and mAP@10: "
            + ", ".join(f"`{profile}`" for profile in tied_with_baseline)
            + "."
        )
    if lower_than_baseline:
        lines.append(
            "Heavier multimodal fusion does not improve this ActivityNet subset and slightly lowers mAP for: "
            + ", ".join(f"`{profile}`" for profile in lower_than_baseline)
            + "."
        )
    lines.extend(
        [
            "These results support framing ActivityNet evaluation as component-level temporal grounding evidence, with ASR/audio fusion reported as an exploratory ablation rather than the main claim.",
            "",
            "## Paper-Ready Sentence",
            "",
            (
                "On the ActivityNet Captions dev200 subset, the visual-only profile achieved "
                f"Recall@10 = {_format_metric(baseline[recall_key])} and "
                f"mAP@10 = {_format_metric(baseline[map_key])} at tIoU >= "
                f"{baseline['tiou_threshold']}; adding ASR/audio evidence did not improve recall "
                "and only matched or slightly reduced mAP in the tested fusion settings."
            ),
            "",
            "## Claim Boundary",
            "",
            "ActivityNet results should not be described as a direct win over prior long-video or lifelog systems unless those systems are evaluated on the same dataset, task, and protocol. Generated-summary or semantic-memory improvements should remain future work until that evidence is indexed, searched, and evaluated in a controlled ablation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _fieldnames(rows: list[dict]) -> list[str]:
    metric_keys = sorted(
        {key for row in rows for key in row if key.startswith("Recall@") or key.startswith("mAP@")}
    )
    return ["profile", "num_videos", "num_queries", "top_k", "tiou_threshold", *metric_keys]


def _metric_keys(rows: list[dict]) -> tuple[str, str]:
    recall_keys = sorted({key for row in rows for key in row if key.startswith("Recall@")})
    map_keys = sorted({key for row in rows for key in row if key.startswith("mAP@")})
    if len(recall_keys) != 1 or len(map_keys) != 1:
        raise ValueError("Expected exactly one Recall@K metric and one mAP@K metric")
    return recall_keys[0], map_keys[0]


def _find_profile(rows: list[dict], profile: str) -> dict:
    for row in rows:
        if row["profile"] == profile:
            return row
    raise ValueError(f"Baseline profile not found: {profile}")


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else str(value)


def _format_metric(value: object) -> str:
    return f"{float(value):.6f}"


def _format_profile_list(profiles: list[str]) -> str:
    return ", ".join(f"`{profile}`" for profile in profiles)


def _latex_escape(value: str) -> str:
    return value.replace("_", "\\_")


if __name__ == "__main__":
    typer.run(main)
