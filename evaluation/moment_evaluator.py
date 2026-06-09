from __future__ import annotations

import asyncio
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
) -> dict:
    profile = get_evaluation_profile(profile_name)
    if profile.tiou_threshold is None:
        raise ValueError(f"Profile '{profile.name}' has no tIoU threshold for evaluation")

    videos = load_evaluation_manifest(manifest_path)
    queries = iter_evaluation_queries(videos)
    duration_by_media_id = {video.media_id: video.duration_sec for video in videos}
    effective_top_k = top_k or profile.top_k
    service = searcher or MomentSearchService()

    results_by_query_id: dict[str, list[RetrievedMoment]] = {}
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

    return {
        "profile": profile.name,
        "top_k": effective_top_k,
        "tiou_threshold": profile.tiou_threshold,
        "num_videos": len(videos),
        "num_queries": len(queries),
        f"Recall@{effective_top_k}": recall_at_k(
            queries,
            results_by_query_id,
            k=effective_top_k,
            tiou_threshold=profile.tiou_threshold,
        ),
        f"mAP@{effective_top_k}": mean_average_precision_at_k(
            queries,
            results_by_query_id,
            k=effective_top_k,
            tiou_threshold=profile.tiou_threshold,
        ),
    }


def main(
    manifest_path: Path = typer.Argument(..., help="Evaluation Manifest JSONL path."),
    profile_name: str = typer.Option(
        "activitynet_visual_heavy",
        help="Evaluation Profile name.",
    ),
    top_k: int | None = typer.Option(None, help="Override profile Top-K."),
) -> None:
    scores = asyncio.run(
        run_moment_evaluation(
            manifest_path=manifest_path,
            profile_name=profile_name,
            top_k=top_k,
        )
    )
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    typer.run(main)
