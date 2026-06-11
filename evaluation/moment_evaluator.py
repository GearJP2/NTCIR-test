from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Protocol

import typer

from app.schemas.search import MomentSearchRequest, MomentSearchResponse
from evaluation.manifest import iter_evaluation_queries, load_evaluation_manifest
from evaluation.profiles import get_evaluation_profile
from evaluation.temporal_metrics import (
    RetrievedMoment,
    mean_average_precision_at_k,
    recall_at_k,
    temporal_iou,
)
from services.moment_search import MomentSearchService


class MomentSearcher(Protocol):
    async def run(self, request: MomentSearchRequest) -> MomentSearchResponse:
        ...


async def run_moment_evaluation(
    manifest_path: Path,
    profile_name: str = "activitynet_visual_heavy",
    top_k: int | None = None,
    searcher: MomentSearcher | None = None,
    summary_path: Path | None = None,
    results_path: Path | None = None,
    query_csv_path: Path | None = None,
    report_path: Path | None = None,
) -> dict:
    profile = get_evaluation_profile(profile_name)
    if profile.tiou_threshold is None:
        raise ValueError(f"Profile '{profile.name}' has no tIoU threshold for evaluation")
    tiou_threshold = profile.tiou_threshold

    videos = load_evaluation_manifest(manifest_path)
    queries = iter_evaluation_queries(videos)
    duration_by_media_id = {video.media_id: video.duration_sec for video in videos}
    effective_top_k = top_k or profile.top_k
    service = searcher or MomentSearchService()

    results_by_query_id: dict[str, list[RetrievedMoment]] = {}
    query_reports: list[dict] = []
    for query in queries:
        response = await service.run(
            MomentSearchRequest(
                media_id=query.media_id,
                query=query.query,
                top_k=effective_top_k,
                duration_sec=duration_by_media_id.get(query.media_id),
                profile=profile.name,
            )
        )
        results_by_query_id[query.query_id] = [
            RetrievedMoment(
                media_id=moment.media_id,
                start_sec=moment.start_sec,
                end_sec=moment.end_sec,
                score=moment.score,
                moment_id=moment.moment_id,
            )
            for moment in response.results
        ]
        ranked_results = results_by_query_id[query.query_id]
        result_reports = []
        for rank, result in enumerate(ranked_results[:effective_top_k], start=1):
            tiou = temporal_iou(query.ground_truth, result)
            result_reports.append(
                {
                    "rank": rank,
                    "moment_id": result.moment_id,
                    "media_id": result.media_id,
                    "start_sec": result.start_sec,
                    "end_sec": result.end_sec,
                    "score": result.score,
                    "tiou": tiou,
                    "hit": tiou >= tiou_threshold,
                }
            )
        hit_ranks = [item["rank"] for item in result_reports if item["hit"]]
        best_tiou = max((item["tiou"] for item in result_reports), default=0.0)
        query_reports.append(
            {
                "query_id": query.query_id,
                "media_id": query.media_id,
                "query": query.query,
                "ground_truth": {
                    "start_sec": query.ground_truth.start_sec,
                    "end_sec": query.ground_truth.end_sec,
                },
                "top_k": effective_top_k,
                "tiou_threshold": tiou_threshold,
                "hit": bool(hit_ranks),
                "hit_rank": hit_ranks[0] if hit_ranks else None,
                "best_tiou": best_tiou,
                "results": result_reports,
            }
        )

    scores = {
        "profile": profile.name,
        "top_k": effective_top_k,
        "tiou_threshold": tiou_threshold,
        "num_videos": len(videos),
        "num_queries": len(queries),
        f"Recall@{effective_top_k}": recall_at_k(
            queries,
            results_by_query_id,
            k=effective_top_k,
            tiou_threshold=tiou_threshold,
        ),
        f"mAP@{effective_top_k}": mean_average_precision_at_k(
            queries,
            results_by_query_id,
            k=effective_top_k,
            tiou_threshold=tiou_threshold,
        ),
    }
    if summary_path is not None:
        _write_json(summary_path, scores)
    if results_path is not None:
        _write_jsonl(results_path, query_reports)
    if query_csv_path is not None:
        _write_query_csv(query_csv_path, query_reports)
    if report_path is not None:
        _write_markdown_report(report_path, scores, query_reports)

    return scores


def main(
    manifest_path: Path = typer.Argument(..., help="Evaluation Manifest JSONL path."),
    profile_name: str = typer.Option(
        "activitynet_visual_heavy",
        help="Evaluation Profile name.",
    ),
    top_k: int | None = typer.Option(None, help="Override profile Top-K."),
    summary_path: Path | None = typer.Option(
        None,
        help="Optional JSON path for aggregate evaluation metrics.",
    ),
    results_path: Path | None = typer.Option(
        None,
        help="Optional JSONL path for per-query Top-K results with scores and tIoU.",
    ),
    query_csv_path: Path | None = typer.Option(
        None,
        help="Optional CSV path for a compact per-query hit/miss table.",
    ),
    report_path: Path | None = typer.Option(
        None,
        help="Optional Markdown path for a human-readable evaluation report.",
    ),
) -> None:
    scores = asyncio.run(
        run_moment_evaluation(
            manifest_path=manifest_path,
            profile_name=profile_name,
            top_k=top_k,
            summary_path=summary_path,
            results_path=results_path,
            query_csv_path=query_csv_path,
            report_path=report_path,
        )
    )
    print(json.dumps(scores, indent=2, sort_keys=True))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_query_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "media_id",
        "hit",
        "hit_rank",
        "best_tiou",
        "gt_start_sec",
        "gt_end_sec",
        "top1_start_sec",
        "top1_end_sec",
        "top1_score",
        "top1_tiou",
        "top1_hit",
        "query",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            top1 = row["results"][0] if row["results"] else {}
            writer.writerow(
                {
                    "query_id": row["query_id"],
                    "media_id": row["media_id"],
                    "hit": row["hit"],
                    "hit_rank": row["hit_rank"],
                    "best_tiou": _round_metric(row["best_tiou"]),
                    "gt_start_sec": row["ground_truth"]["start_sec"],
                    "gt_end_sec": row["ground_truth"]["end_sec"],
                    "top1_start_sec": top1.get("start_sec"),
                    "top1_end_sec": top1.get("end_sec"),
                    "top1_score": _round_metric(top1.get("score")),
                    "top1_tiou": _round_metric(top1.get("tiou")),
                    "top1_hit": top1.get("hit"),
                    "query": row["query"],
                }
            )


def _write_markdown_report(path: Path, scores: dict, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    recall_key = f"Recall@{scores['top_k']}"
    map_key = f"mAP@{scores['top_k']}"
    hits = sum(1 for row in rows if row["hit"])
    misses = len(rows) - hits
    hit_rank_counts = {
        rank: sum(1 for row in rows if row["hit_rank"] == rank)
        for rank in range(1, scores["top_k"] + 1)
    }
    miss_examples = sorted(
        (row for row in rows if not row["hit"]),
        key=lambda row: row["best_tiou"],
        reverse=True,
    )[:10]

    lines = [
        "# ActivityNet Moment Search Evaluation",
        "",
        "## Summary",
        "",
        f"- Profile: `{scores['profile']}`",
        f"- Videos: {scores['num_videos']}",
        f"- Queries: {scores['num_queries']}",
        f"- Top-K: {scores['top_k']}",
        f"- tIoU threshold: {scores['tiou_threshold']}",
        f"- {recall_key}: {_round_metric(scores[recall_key])}",
        f"- {map_key}: {_round_metric(scores[map_key])}",
        f"- Hits: {hits}",
        f"- Misses: {misses}",
        "",
        "## Hit Rank Distribution",
        "",
        "| Rank | Hits |",
        "| ---: | ---: |",
        *[f"| {rank} | {count} |" for rank, count in hit_rank_counts.items() if count],
        "",
        "## Closest Misses",
        "",
        "| Query ID | Best tIoU | GT Moment | Top-1 Moment | Query |",
        "| --- | ---: | --- | --- | --- |",
        *[_format_miss_row(row) for row in miss_examples],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_miss_row(row: dict) -> str:
    top1 = row["results"][0] if row["results"] else {}
    gt = row["ground_truth"]
    query = str(row["query"]).replace("|", "\\|")
    return (
        f"| `{row['query_id']}` | {_round_metric(row['best_tiou'])} | "
        f"{gt['start_sec']:.2f}-{gt['end_sec']:.2f}s | "
        f"{top1.get('start_sec', 0.0):.2f}-{top1.get('end_sec', 0.0):.2f}s | "
        f"{query} |"
    )


def _round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


if __name__ == "__main__":
    typer.run(main)
