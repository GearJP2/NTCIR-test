"""
Unit tests for QueryService, WorldMMPromptBuilder, and the /search/episodic endpoint.
No real LLM, Milvus, or MinIO is required — all external calls are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.schemas.search import EpisodicHit, EpisodicSearchRequest, LLMReasoning
from services.query_service import (
    WorldMMPromptBuilder,
    _clamp,
    _noop_reasoning,
    _parse_llm_json,
    _seconds_to_timestamp,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_hit(**kwargs) -> EpisodicHit:
    defaults = dict(
        chunk_id="chunk-abc",
        media_id="media-1",
        score=0.87,
        start_sec=12.5,
        end_sec=42.1,
        duration_sec=29.6,
        transcript="The team discussed the machine learning pipeline in detail.",
        language="en",
        minio_url="http://minio/bucket/chunks/media-1/chunk-abc.wav",
        object_key="chunks/media-1/chunk-abc.wav",
        embedding_model="clap",
        created_at=1_700_000_000,
    )
    defaults.update(kwargs)
    return EpisodicHit(**defaults)


# ── Helper utilities ──────────────────────────────────────────────────────────

class TestHelpers:
    def test_seconds_to_timestamp_zero(self):
        assert _seconds_to_timestamp(0.0) == "00:00:00.000"

    def test_seconds_to_timestamp_with_hours(self):
        assert _seconds_to_timestamp(3723.5) == "01:02:03.500"

    def test_clamp_within_range(self):
        assert _clamp(0.7) == 0.7

    def test_clamp_below_zero(self):
        assert _clamp(-0.1) == 0.0

    def test_clamp_above_one(self):
        assert _clamp(1.5) == 1.0


# ── WorldMMPromptBuilder ──────────────────────────────────────────────────────

class TestWorldMMPromptBuilder:
    def test_build_returns_valid_json(self):
        hits = [_make_hit()]
        prompt_json = WorldMMPromptBuilder.build("Find the ML project discussion", hits)
        data = json.loads(prompt_json)
        assert "system" in data
        assert "user" in data

    def test_prompt_contains_query(self):
        hits = [_make_hit()]
        prompt_json = WorldMMPromptBuilder.build("Find the ML project discussion", hits)
        data = json.loads(prompt_json)
        assert "Find the ML project discussion" in data["user"]

    def test_prompt_contains_chunk_id(self):
        hits = [_make_hit(chunk_id="my-special-chunk")]
        data = json.loads(WorldMMPromptBuilder.build("query", hits))
        assert "my-special-chunk" in data["user"]

    def test_prompt_contains_transcript(self):
        hits = [_make_hit(transcript="Talk about neural networks")]
        data = json.loads(WorldMMPromptBuilder.build("query", hits))
        assert "neural networks" in data["user"]

    def test_prompt_lists_all_hits(self):
        hits = [_make_hit(chunk_id=f"c{i}") for i in range(4)]
        data = json.loads(WorldMMPromptBuilder.build("query", hits))
        for i in range(4):
            assert f"c{i}" in data["user"]

    def test_system_prompt_contains_json_schema(self):
        hits = [_make_hit()]
        data = json.loads(WorldMMPromptBuilder.build("query", hits))
        assert "best_chunk_ids" in data["system"]
        assert "confidence" in data["system"]


# ── _parse_llm_json ───────────────────────────────────────────────────────────

class TestParseLLMJson:
    def test_parse_valid_json(self):
        raw = json.dumps({
            "answer": "The discussion happened at 12.5s.",
            "best_chunk_ids": ["chunk-abc"],
            "reasoning": "Segment 1 contains ML keywords.",
            "confidence": 0.92,
        })
        result = _parse_llm_json(raw, model_used="gpt-4o-mini", hits=[_make_hit()])
        assert result.answer == "The discussion happened at 12.5s."
        assert result.best_chunk_ids == ["chunk-abc"]
        assert result.confidence == pytest.approx(0.92)
        assert result.model_used == "gpt-4o-mini"

    def test_parse_strips_markdown_fences(self):
        raw = "```json\n{\"answer\":\"ok\",\"best_chunk_ids\":[],\"reasoning\":\"r\",\"confidence\":0.5}\n```"
        result = _parse_llm_json(raw, model_used="m", hits=[])
        assert result.answer == "ok"

    def test_parse_invalid_json_returns_fallback(self):
        result = _parse_llm_json("NOT_JSON", model_used="m", hits=[_make_hit()])
        assert result.confidence == 0.0
        assert result.best_chunk_ids == ["chunk-abc"]  # top hit used as fallback

    def test_parse_clamps_confidence_above_one(self):
        raw = json.dumps({"answer":"a","best_chunk_ids":[],"reasoning":"r","confidence":1.8})
        result = _parse_llm_json(raw, model_used="m", hits=[])
        assert result.confidence <= 1.0

    def test_parse_clamps_confidence_below_zero(self):
        raw = json.dumps({"answer":"a","best_chunk_ids":[],"reasoning":"r","confidence":-0.3})
        result = _parse_llm_json(raw, model_used="m", hits=[])
        assert result.confidence >= 0.0


# ── _noop_reasoning ───────────────────────────────────────────────────────────

class TestNoopReasoning:
    def test_noop_uses_top_hits(self):
        hits = [_make_hit(chunk_id=f"c{i}") for i in range(5)]
        result = _noop_reasoning(hits)
        assert result.model_used == "none"
        assert all(cid in result.best_chunk_ids for cid in ["c0", "c1", "c2"])

    def test_noop_no_hits(self):
        result = _noop_reasoning([])
        assert "No results" in result.answer
        assert result.best_chunk_ids == []

    def test_noop_includes_error_message(self):
        result = _noop_reasoning([], error="GPU OOM")
        assert "GPU OOM" in result.reasoning


# ── QueryService (integration-style, all I/O mocked) ─────────────────────────

class TestQueryService:
    def _fake_vector(self, dim: int = 512) -> np.ndarray:
        v = np.random.rand(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    @pytest.mark.asyncio
    @patch("services.query_service._LLMBackend.reason", new_callable=AsyncMock)
    @patch("services.query_service.MilvusService")
    @patch("services.query_service._ClapTextTowerCache.get")
    async def test_run_returns_response(
        self, mock_clap_cache, mock_milvus_cls, mock_llm
    ):
        # Arrange
        fake_vec = self._fake_vector(512)
        processor_mock = MagicMock()
        processor_mock.return_value = {"input_ids": MagicMock()}
        model_mock = MagicMock()
        import torch
        features = torch.tensor([fake_vec])
        features_norm = features / features.norm(dim=-1, keepdim=True)
        model_mock.get_text_features.return_value = features_norm
        mock_clap_cache.return_value = (processor_mock, model_mock, "cpu")

        mock_milvus_instance = mock_milvus_cls.return_value
        mock_milvus_instance.search.return_value = [
            {
                "chunk_id": "c1", "media_id": "m1",
                "score": 0.9, "start_sec": 0.0, "end_sec": 10.0,
                "duration_sec": 10.0, "transcript": "hello world",
                "language": "en", "minio_url": "", "object_key": "",
                "embedding_model": "clap", "created_at": 0,
            }
        ]
        mock_llm.return_value = LLMReasoning(
            answer="Found it", best_chunk_ids=["c1"],
            reasoning="Obvious match", confidence=0.9, model_used="mock",
        )

        # Act
        from services.query_service import QueryService
        with patch("services.query_service._hydrate_hits", side_effect=lambda x: [_make_hit(chunk_id=r["chunk_id"]) for r in x]):
            svc = QueryService(embedder_name="clap")
            resp = await svc.run("Find the ML project discussion", top_k=3)

        # Assert
        assert resp.total_hits == 1
        assert resp.hits[0].chunk_id == "c1"
        assert resp.reasoning is not None
        assert resp.reasoning.answer == "Found it"
        assert resp.embedder_used == "clap"

    @pytest.mark.asyncio
    @patch("services.query_service.MilvusService")
    @patch("services.query_service._ClapTextTowerCache.get")
    async def test_run_without_llm_returns_no_reasoning(
        self, mock_clap_cache, mock_milvus_cls
    ):
        fake_vec = self._fake_vector(512)
        import torch
        features = torch.tensor([fake_vec])
        model_mock = MagicMock()
        model_mock.get_text_features.return_value = features / features.norm(dim=-1, keepdim=True)
        mock_clap_cache.return_value = (MagicMock(), model_mock, "cpu")

        mock_milvus_cls.return_value.search.return_value = []

        from services.query_service import QueryService
        svc = QueryService(embedder_name="clap")
        resp = await svc.run("empty query", top_k=5, use_llm=False)

        assert resp.reasoning is None
        assert resp.total_hits == 0


# ── Search endpoint integration tests ────────────────────────────────────────

class TestSearchEpisodicEndpoint:
    @pytest.mark.asyncio
    @patch("app.api.v1.endpoints.search.QueryService")
    async def test_endpoint_returns_200(self, mock_qs_cls, client):
        from app.schemas.search import EpisodicSearchResponse

        mock_instance = mock_qs_cls.return_value
        mock_instance.run = AsyncMock(return_value=EpisodicSearchResponse(
            query="test query",
            total_hits=1,
            hits=[_make_hit()],
            reasoning=None,
            query_vector_dim=512,
            embedder_used="clap",
        ))

        resp = client.post(
            "/api/v1/search/episodic",
            json={"query": "Find the moment about machine learning", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_hits"] == 1
        assert data["hits"][0]["chunk_id"] == "chunk-abc"

    @pytest.mark.asyncio
    async def test_endpoint_rejects_empty_query(self, client):
        resp = client.post(
            "/api/v1/search/episodic",
            json={"query": "ab"},  # below min_length=3
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_convenience_alias_exists(self, client):
        """The /api/search/episodic path must be registered."""
        resp = client.post(
            "/api/search/episodic",
            json={"query": "test", "use_llm": False},
        )
        # 500 is acceptable here (no real Milvus) — 404 is not
        assert resp.status_code != 404
