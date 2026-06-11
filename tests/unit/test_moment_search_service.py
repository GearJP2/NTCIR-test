import numpy as np
import pytest

from app.schemas.search import MomentSearchRequest
from services.moment_search import MomentSearchService


class FakeVisualEncoder:
    def encode_text(self, texts):
        assert texts == ["woman doing sit ups"]
        return [np.ones(768, dtype=np.float32)]


class FailingVisualEncoder:
    def encode_text(self, texts):
        raise RuntimeError("clip unavailable")


class FakeTextEncoder:
    def encode(self, text):
        assert text == "woman doing sit ups"
        return np.ones(768, dtype=np.float32) * 2


class FakeClapEncoder:
    def encode_text(self, texts):
        assert texts == ["woman doing sit ups"]
        return [np.ones(512, dtype=np.float32)]


class FakeMilvusResult:
    def __init__(self, entity, score):
        self.entity = entity
        self.score = score


class FakeMilvus:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["collection_name"] == "visual_keyframes":
            return [[
                FakeMilvusResult(
                    {
                        "frame_id": "frame-12",
                        "media_id": "v_123",
                        "timestamp_sec": 12.0,
                    },
                    0.8,
                ),
                FakeMilvusResult(
                    {
                        "frame_id": "frame-22",
                        "media_id": "v_123",
                        "timestamp_sec": 22.0,
                    },
                    0.6,
                ),
            ]]

        if kwargs["collection_name"] == "text_transcripts":
            return [[
                FakeMilvusResult(
                    {
                        "chunk_id": "asr-1",
                        "media_id": "v_123",
                        "start_sec": 16.0,
                        "end_sec": 24.0,
                        "text": "the woman talks about doing sit ups",
                    },
                    2.0,
                )
            ]]

        if kwargs["collection_name"] == "audio_segments":
            return [[
                FakeMilvusResult(
                    {
                        "segment_id": "audio-1",
                        "media_id": "v_123",
                        "start_sec": 14.0,
                        "end_sec": 19.0,
                        "summary": "audio resembles exercise activity",
                    },
                    1.5,
                )
            ]]

        raise AssertionError(f"unexpected collection {kwargs['collection_name']}")


@pytest.mark.asyncio
async def test_moment_search_service_returns_visual_video_moments(monkeypatch):
    fake_milvus = FakeMilvus()
    monkeypatch.setattr(
        "services.embedding.visual_encoder.VisualEncoder",
        lambda: FakeVisualEncoder(),
    )
    monkeypatch.setattr(
        "services.embedding.text_encoder.TextEncoder",
        lambda: FakeTextEncoder(),
    )
    monkeypatch.setattr(
        "services.embedding.clap_encoder.ClapEncoder",
        lambda: FakeClapEncoder(),
    )
    monkeypatch.setattr(
        "storage.milvus.client.get_milvus_client",
        lambda: fake_milvus,
    )

    response = await MomentSearchService().run(
        MomentSearchRequest(
            media_id="v_123",
            query="woman doing sit ups",
            top_k=3,
            duration_sec=30.0,
            profile="activitynet_visual_heavy",
        )
    )

    assert response.total == 3
    assert response.results[0].moment_id == "v_123:10.000-20.000"
    assert response.results[0].score == 0.5
    evidence_by_source = {
        evidence.source_type: evidence for evidence in response.results[0].evidence
    }
    assert evidence_by_source["visual"].source_id == "frame-12"
    assert evidence_by_source["asr"].source_id == "asr-1"
    assert evidence_by_source["asr"].text == "the woman talks about doing sit ups"
    assert evidence_by_source["audio"].source_id == "audio-1"
    assert evidence_by_source["audio"].text == "audio resembles exercise activity"

    visual_call = fake_milvus.calls[0]
    text_call = fake_milvus.calls[1]
    audio_call = fake_milvus.calls[2]
    assert visual_call["collection_name"] == "visual_keyframes"
    assert visual_call["filter"] == 'media_id == "v_123"'
    assert visual_call["limit"] == 25
    assert text_call["collection_name"] == "text_transcripts"
    assert text_call["filter"] == 'media_id == "v_123"'
    assert text_call["limit"] == 25
    assert audio_call["collection_name"] == "audio_segments"
    assert audio_call["filter"] == 'media_id == "v_123"'
    assert audio_call["limit"] == 25


@pytest.mark.asyncio
async def test_moment_search_service_returns_empty_without_duration(monkeypatch):
    def fail_visual_encoder():
        raise AssertionError("visual encoder should not be loaded without duration")

    monkeypatch.setattr("services.embedding.visual_encoder.VisualEncoder", fail_visual_encoder)
    monkeypatch.setattr("services.embedding.text_encoder.TextEncoder", fail_visual_encoder)
    monkeypatch.setattr("services.embedding.clap_encoder.ClapEncoder", fail_visual_encoder)

    response = await MomentSearchService().run(
        MomentSearchRequest(
            media_id="v_123",
            query="woman doing sit ups",
            top_k=3,
            profile="activitynet_visual_heavy",
        )
    )

    assert response.results == []
    assert response.total == 0


@pytest.mark.asyncio
async def test_moment_search_service_skips_zero_weight_modalities(monkeypatch):
    fake_milvus = FakeMilvus()
    monkeypatch.setattr(
        "services.embedding.visual_encoder.VisualEncoder",
        lambda: FakeVisualEncoder(),
    )

    def fail_unused_encoder():
        raise AssertionError("zero-weight encoder should not be loaded")

    monkeypatch.setattr("services.embedding.text_encoder.TextEncoder", fail_unused_encoder)
    monkeypatch.setattr("services.embedding.clap_encoder.ClapEncoder", fail_unused_encoder)
    monkeypatch.setattr(
        "storage.milvus.client.get_milvus_client",
        lambda: fake_milvus,
    )

    response = await MomentSearchService().run(
        MomentSearchRequest(
            media_id="v_123",
            query="woman doing sit ups",
            top_k=3,
            duration_sec=30.0,
            profile="activitynet_visual_only",
        )
    )

    assert response.total == 3
    assert [call["collection_name"] for call in fake_milvus.calls] == ["visual_keyframes"]


@pytest.mark.asyncio
async def test_moment_search_service_continues_when_one_modality_fails(monkeypatch):
    fake_milvus = FakeMilvus()
    monkeypatch.setattr(
        "services.embedding.visual_encoder.VisualEncoder",
        lambda: FailingVisualEncoder(),
    )
    monkeypatch.setattr(
        "services.embedding.text_encoder.TextEncoder",
        lambda: FakeTextEncoder(),
    )
    monkeypatch.setattr(
        "services.embedding.clap_encoder.ClapEncoder",
        lambda: FakeClapEncoder(),
    )
    monkeypatch.setattr(
        "storage.milvus.client.get_milvus_client",
        lambda: fake_milvus,
    )

    response = await MomentSearchService().run(
        MomentSearchRequest(
            media_id="v_123",
            query="woman doing sit ups",
            top_k=3,
            duration_sec=30.0,
            profile="activitynet_visual_heavy",
        )
    )

    assert response.total == 3
    evidence_sources = {
        evidence.source_type
        for moment in response.results
        for evidence in moment.evidence
    }
    assert "visual" not in evidence_sources
    assert {"asr", "audio"}.issubset(evidence_sources)
