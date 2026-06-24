from __future__ import annotations

import csv
import json
from pathlib import Path

import typer


def compare_results(
    baseline_results_path: Path,
    candidate_results_path: Path,
    csv_path: Path,
    markdown_path: Path,
    json_path: Path | None = None,
    limit: int = 50,
) -> dict:
    baseline = _load_query_results(baseline_results_path)
    candidate = _load_query_results(candidate_results_path)
    missing = sorted(set(baseline) ^ set(candidate))
    if missing:
        raise ValueError(f"Result files have mismatched query IDs: {missing[:10]}")

    rows = [_compare_query(baseline[query_id], candidate[query_id]) for query_id in sorted(baseline)]
    summary = {
        "baseline_profile": _profile_from_path(baseline_results_path),
        "candidate_profile": _profile_from_path(candidate_results_path),
        "num_queries": len(rows),
        "baseline_hits": sum(1 for row in rows if row["baseline_hit"]),
        "candidate_hits": sum(1 for row in rows if row["candidate_hit"]),
        "regressions": sum(1 for row in rows if row["change_type"] == "regression"),
        "improvements": sum(1 for row in rows if row["change_type"] == "improvement"),
        "rank_improvements": sum(1 for row in rows if row["change_type"] == "rank_improvement"),
        "rank_regressions": sum(1 for row in rows if row["change_type"] == "rank_regression"),
        "unchanged": sum(1 for row in rows if row["change_type"] == "unchanged"),
    }
    _write_csv(csv_path, rows)
    _write_markdown(markdown_path, summary, rows, limit=limit)
    if json_path is not None:
        _write_json(json_path, {"summary": summary, "queries": rows})
    return {"summary": summary, "queries": rows}


def main(
    baseline_results_path: Path = typer.Option(
        ...,
        help="Baseline evaluator JSONL results path.",
    ),
    candidate_results_path: Path = typer.Option(
        ...,
        help="Candidate evaluator JSONL results path.",
    ),
    csv_path: Path = typer.Option(
        Path("data/evaluation/activitynet_regressions.csv"),
        help="Output per-query comparison CSV path.",
    ),
    markdown_path: Path = typer.Option(
        Path("data/evaluation/activitynet_regressions.md"),
        help="Output Markdown comparison report path.",
    ),
    json_path: Path | None = typer.Option(
        None,
        help="Optional JSON output path.",
    ),
    limit: int = typer.Option(50, help="Maximum rows per Markdown section."),
) -> None:
    result = compare_results(
        baseline_results_path=baseline_results_path,
        candidate_results_path=candidate_results_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
        json_path=json_path,
        limit=limit,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


def _load_query_results(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["query_id"]] = row
    return rows


def _compare_query(baseline: dict, candidate: dict) -> dict:
    baseline_hit = bool(baseline["hit"])
    candidate_hit = bool(candidate["hit"])
    baseline_rank = baseline.get("hit_rank")
    candidate_rank = candidate.get("hit_rank")
    baseline_top1 = _top_result(baseline)
    candidate_top1 = _top_result(candidate)

    if baseline_hit and not candidate_hit:
        change_type = "regression"
    elif candidate_hit and not baseline_hit:
        change_type = "improvement"
    elif baseline_hit and candidate_hit and candidate_rank < baseline_rank:
        change_type = "rank_improvement"
    elif baseline_hit and candidate_hit and candidate_rank > baseline_rank:
        change_type = "rank_regression"
    else:
        change_type = "unchanged"

    return {
        "query_id": baseline["query_id"],
        "media_id": baseline["media_id"],
        "change_type": change_type,
        "baseline_hit": baseline_hit,
        "candidate_hit": candidate_hit,
        "baseline_hit_rank": baseline_rank,
        "candidate_hit_rank": candidate_rank,
        "baseline_best_tiou": baseline["best_tiou"],
        "candidate_best_tiou": candidate["best_tiou"],
        "best_tiou_delta": candidate["best_tiou"] - baseline["best_tiou"],
        "baseline_top1": _format_moment(baseline_top1),
        "candidate_top1": _format_moment(candidate_top1),
        "ground_truth": _format_ground_truth(baseline),
        "query": baseline["query"],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "media_id",
        "change_type",
        "baseline_hit",
        "candidate_hit",
        "baseline_hit_rank",
        "candidate_hit_rank",
        "baseline_best_tiou",
        "candidate_best_tiou",
        "best_tiou_delta",
        "ground_truth",
        "baseline_top1",
        "candidate_top1",
        "query",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict, rows: list[dict], limit: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Moment Search Regression Report",
        "",
        "## Summary",
        "",
        f"- Baseline: `{summary['baseline_profile']}`",
        f"- Candidate: `{summary['candidate_profile']}`",
        f"- Queries: {summary['num_queries']}",
        f"- Baseline hits: {summary['baseline_hits']}",
        f"- Candidate hits: {summary['candidate_hits']}",
        f"- Regressions: {summary['regressions']}",
        f"- Improvements: {summary['improvements']}",
        f"- Rank regressions: {summary['rank_regressions']}",
        f"- Rank improvements: {summary['rank_improvements']}",
        "",
    ]
    for change_type, title in [
        ("regression", "Lost Hits"),
        ("improvement", "Gained Hits"),
        ("rank_regression", "Worse Hit Rank"),
        ("rank_improvement", "Better Hit Rank"),
    ]:
        section_rows = [row for row in rows if row["change_type"] == change_type]
        lines.extend(_section(title, section_rows[:limit]))
    path.write_text("\n".join(lines), encoding="utf-8")


def _section(title: str, rows: list[dict]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Query ID | Baseline | Candidate | Best tIoU Delta | GT | Query |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    if not rows:
        return [*lines, "|  |  |  |  |  |  |", ""]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['query_id']}`",
                    _format_hit(row["baseline_hit"], row["baseline_hit_rank"]),
                    _format_hit(row["candidate_hit"], row["candidate_hit_rank"]),
                    f"{row['best_tiou_delta']:.6f}",
                    row["ground_truth"],
                    str(row["query"]).replace("|", "\\|"),
                ]
            )
            + " |"
        )
    return [*lines, ""]


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _top_result(row: dict) -> dict | None:
    return row["results"][0] if row["results"] else None


def _format_moment(moment: dict | None) -> str:
    if moment is None:
        return ""
    return f"{moment['start_sec']:.2f}-{moment['end_sec']:.2f}s"


def _format_ground_truth(row: dict) -> str:
    gt = row["ground_truth"]
    return f"{gt['start_sec']:.2f}-{gt['end_sec']:.2f}s"


def _format_hit(hit: bool, rank: int | None) -> str:
    return f"hit@{rank}" if hit else "miss"


def _profile_from_path(path: Path) -> str:
    name = path.stem
    return name.removesuffix("_results")


if __name__ == "__main__":
    typer.run(main)
