import json

import pytest

from app.schemas.search import Evidence, MomentSearchRequest, MomentSearchResponse, VideoMoment
from evaluation.castle_inspection import run_castle_inspection


class FakeSearcher:
    def __init__(self):
        self.requests = []

    async def run(self, request: MomentSearchRequest) -> MomentSearchResponse:
        self.requests.append(request)
        return MomentSearchResponse(
            media_id=request.media_id,
            query=request.query,
            top_k=request.top_k,
            profile=request.profile,
            results=[
                VideoMoment(
                    rank=1,
                    moment_id=f"{request.media_id}:10.000-20.000",
                    media_id=request.media_id,
                    start_sec=10.0,
                    end_sec=20.0,
                    score=0.7,
                    thumbnail_sec=15.0,
                    evidence=[
                        Evidence(
                            source_type="visual",
                            score=0.7,
                            source_id="frame-1",
                            timestamp_sec=14.0,
                        )
                    ],
                )
            ],
            total=1,
        )


@pytest.mark.asyncio
async def test_run_castle_inspection_writes_jsonl(tmp_path):
    queries_path = tmp_path / "queries.jsonl"
    output_path = tmp_path / "inspection.jsonl"
    queries_path.write_text(
        json.dumps(
            {
                "query_id": "castle_smoke:001",
                "query": "person eating a meal",
                "tags": ["food"],
                "notes": "Basic food scene.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    searcher = FakeSearcher()

    rows = await run_castle_inspection(
        queries_path=queries_path,
        media_id="castle_001",
        duration_sec=120.0,
        output_path=output_path,
        searcher=searcher,
    )

    assert len(rows) == 1
    assert searcher.requests[0].media_id == "castle_001"
    assert searcher.requests[0].duration_sec == 120.0
    assert searcher.requests[0].profile == "castle_lifelog_balanced"

    output_row = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_row["query_id"] == "castle_smoke:001"
    assert output_row["results"][0]["moment_id"] == "castle_001:10.000-20.000"
    assert output_row["results"][0]["evidence"][0]["source_type"] == "visual"


@pytest.mark.asyncio
async def test_run_castle_inspection_query_media_id_overrides_default(tmp_path):
    queries_path = tmp_path / "queries.jsonl"
    output_path = tmp_path / "inspection.jsonl"
    queries_path.write_text(
        json.dumps(
            {
                "query_id": "castle_smoke:001",
                "query": "person eating a meal",
                "media_id": "castle_pinned",
                "tags": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    searcher = FakeSearcher()

    await run_castle_inspection(
        queries_path=queries_path,
        media_id="castle_default",
        duration_sec=120.0,
        output_path=output_path,
        searcher=searcher,
    )

    assert searcher.requests[0].media_id == "castle_pinned"
