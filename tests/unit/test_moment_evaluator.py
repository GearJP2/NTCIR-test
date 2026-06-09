import json

import pytest

from app.schemas.search import MomentSearchRequest, MomentSearchResponse, VideoMoment
from evaluation.moment_evaluator import run_moment_evaluation


class FakeSearcher:
    async def run(self, request: MomentSearchRequest) -> MomentSearchResponse:
        if "sit ups" in request.query:
            results = [
                VideoMoment(
                    rank=1,
                    moment_id=f"{request.media_id}:40.000-50.000",
                    media_id=request.media_id,
                    start_sec=40.0,
                    end_sec=50.0,
                    score=0.9,
                    thumbnail_sec=45.0,
                    evidence=[],
                )
            ]
        else:
            results = [
                VideoMoment(
                    rank=1,
                    moment_id=f"{request.media_id}:0.000-10.000",
                    media_id=request.media_id,
                    start_sec=0.0,
                    end_sec=10.0,
                    score=0.9,
                    thumbnail_sec=5.0,
                    evidence=[],
                )
            ]

        return MomentSearchResponse(
            media_id=request.media_id,
            query=request.query,
            top_k=request.top_k,
            profile=request.profile,
            results=results,
            total=len(results),
        )


@pytest.mark.asyncio
async def test_run_moment_evaluation_reports_temporal_metrics(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    row = {
        "media_id": "v_123",
        "video_path": "videos/v_123.mp4",
        "duration_sec": 120.0,
        "queries": [
            {
                "query_id": "v_123:0",
                "query": "A woman is doing sit ups",
                "ground_truth": {"start_sec": 39.0, "end_sec": 51.0},
            },
            {
                "query_id": "v_123:1",
                "query": "A woman rests",
                "ground_truth": {"start_sec": 80.0, "end_sec": 90.0},
            },
        ],
    }
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    scores = await run_moment_evaluation(
        manifest_path=manifest_path,
        searcher=FakeSearcher(),
    )

    assert scores["profile"] == "activitynet_visual_heavy"
    assert scores["top_k"] == 10
    assert scores["tiou_threshold"] == 0.3
    assert scores["num_videos"] == 1
    assert scores["num_queries"] == 2
    assert scores["Recall@10"] == 0.5
    assert scores["mAP@10"] == 0.5


@pytest.mark.asyncio
async def test_run_moment_evaluation_rejects_profile_without_tiou(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "media_id": "v_123",
                "video_path": "videos/v_123.mp4",
                "queries": [
                    {
                        "query": "A woman is doing sit ups",
                        "ground_truth": {"start_sec": 39.0, "end_sec": 51.0},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no tIoU threshold"):
        await run_moment_evaluation(
            manifest_path=manifest_path,
            profile_name="castle_lifelog_balanced",
            searcher=FakeSearcher(),
        )
