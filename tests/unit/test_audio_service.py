"""
Unit tests for AudioService chunking strategies.
Embedding and storage calls are mocked so no GPU/Milvus/MinIO is required.
"""

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from services.audio_service import (
    AudioChunk,
    AudioService,
    ChunkingStrategy,
    _save_chunk,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def silent_wav(tmp_path) -> Path:
    """1-minute silent 16 kHz mono WAV."""
    path = tmp_path / "silence.wav"
    sf.write(str(path), np.zeros(16_000 * 60, dtype=np.float32), 16_000)
    return path


@pytest.fixture
def short_wav(tmp_path) -> Path:
    """0.5-second silent WAV (below min_chunk_sec=1.0 → should produce no chunks)."""
    path = tmp_path / "short.wav"
    sf.write(str(path), np.zeros(8_000, dtype=np.float32), 16_000)
    return path


def _dummy_embed_batch(paths: list[str]) -> list[np.ndarray]:
    """Return random 512-dim unit vectors without loading any real model."""
    vecs = [np.random.rand(512).astype(np.float32) for _ in paths]
    return [v / np.linalg.norm(v) for v in vecs]


# ── _save_chunk ───────────────────────────────────────────────────────────────

def test_save_chunk_creates_file(tmp_path):
    samples = np.zeros(16_000, dtype=np.float32)
    chunk = _save_chunk(samples, 16_000, 0.0, 1.0, "m1", tmp_path, "th")
    assert Path(chunk.local_path).exists()
    assert chunk.duration_sec == pytest.approx(1.0, abs=0.01)
    assert chunk.language == "th"
    assert len(chunk.chunk_id) == 36  # UUID


# ── FIXED_DURATION strategy ───────────────────────────────────────────────────

@patch("services.audio_service._get_embedder")
def test_fixed_duration_produces_chunks(mock_get_embedder, silent_wav, tmp_path):
    mock_embedder = MagicMock()
    mock_embedder.embed_batch.side_effect = _dummy_embed_batch
    mock_get_embedder.return_value = mock_embedder

    service = AudioService(
        strategy=ChunkingStrategy.FIXED_DURATION,
        embedder_name="clap",
        chunk_duration_sec=15.0,
        min_chunk_sec=1.0,
    )
    chunks = service.process_file(silent_wav, "test-media", tmp_dir=tmp_path)

    assert len(chunks) > 0
    for c in chunks:
        assert c.duration_sec >= 1.0
        assert c.embedding.shape == (512,)
        assert abs(np.linalg.norm(c.embedding) - 1.0) < 1e-4


@patch("services.audio_service._get_embedder")
def test_fixed_duration_no_overlap_past_end(mock_get_embedder, silent_wav, tmp_path):
    mock_embedder = MagicMock()
    mock_embedder.embed_batch.side_effect = _dummy_embed_batch
    mock_get_embedder.return_value = mock_embedder

    service = AudioService(
        strategy=ChunkingStrategy.FIXED_DURATION,
        chunk_duration_sec=20.0,
    )
    chunks = service.process_file(silent_wav, "m2", tmp_dir=tmp_path)
    # No chunk should exceed the file duration
    total_dur = 60.0
    for c in chunks:
        assert c.end_sec <= total_dur + 0.1


# ── VAD strategy ──────────────────────────────────────────────────────────────

@patch("services.audio_service._get_embedder")
def test_vad_silence_produces_no_chunks(mock_get_embedder, silent_wav, tmp_path):
    """Pure silence should produce zero speech segments."""
    mock_embedder = MagicMock()
    mock_embedder.embed_batch.side_effect = _dummy_embed_batch
    mock_get_embedder.return_value = mock_embedder

    service = AudioService(strategy=ChunkingStrategy.VAD)
    chunks = service.process_file(silent_wav, "m3", tmp_dir=tmp_path)
    # All frames are silent → no speech segments
    assert chunks == []


@patch("services.audio_service._get_embedder")
def test_short_file_below_min_chunk(mock_get_embedder, short_wav, tmp_path):
    """Files shorter than min_chunk_sec should produce no chunks."""
    mock_embedder = MagicMock()
    mock_embedder.embed_batch.side_effect = _dummy_embed_batch
    mock_get_embedder.return_value = mock_embedder

    service = AudioService(strategy=ChunkingStrategy.FIXED_DURATION, min_chunk_sec=1.0)
    chunks = service.process_file(short_wav, "m4", tmp_dir=tmp_path)
    assert chunks == []


# ── Error paths ───────────────────────────────────────────────────────────────

def test_process_file_raises_on_missing_file(tmp_path):
    service = AudioService()
    with pytest.raises(FileNotFoundError):
        service.process_file(tmp_path / "nonexistent.wav", "mx")


@patch("services.audio_service._get_embedder")
def test_embedding_error_propagates(mock_get_embedder, silent_wav, tmp_path):
    from app.core.exceptions import EmbeddingError

    mock_embedder = MagicMock()
    mock_embedder.embed_batch.side_effect = RuntimeError("GPU OOM")
    mock_get_embedder.return_value = mock_embedder

    service = AudioService(
        strategy=ChunkingStrategy.FIXED_DURATION,
        chunk_duration_sec=10.0,
    )
    with pytest.raises(EmbeddingError, match="GPU OOM"):
        service.process_file(silent_wav, "m5", tmp_dir=tmp_path)


# ── MilvusService unit tests ─────────────────────────────────────────────────

class TestMilvusService:
    def _make_mock_client(self):
        client = MagicMock()
        client.has_collection.return_value = True  # skip collection creation
        return client

    def _make_chunk(self, **kwargs) -> AudioChunk:
        defaults = dict(
            chunk_id=str(uuid.uuid4()),
            media_id="media-1",
            start_sec=0.0,
            end_sec=5.0,
            duration_sec=5.0,
            local_path="/tmp/fake.wav",
            embedding=np.random.rand(512).astype(np.float32),
        )
        defaults.update(kwargs)
        return AudioChunk(**defaults)

    def test_upsert_returns_count(self):
        from storage.milvus.milvus_service import MilvusService

        client = self._make_mock_client()
        client.upsert.return_value = {"upsert_count": 3}
        svc = MilvusService(client=client)

        chunks = [self._make_chunk() for _ in range(3)]
        result = svc.upsert_chunks(chunks)
        assert result == 3
        client.upsert.assert_called_once()

    def test_upsert_skips_empty_embedding(self):
        from storage.milvus.milvus_service import MilvusService

        client = self._make_mock_client()
        client.upsert.return_value = {"upsert_count": 1}
        svc = MilvusService(client=client)

        good = self._make_chunk()
        bad = self._make_chunk(embedding=np.array([], dtype=np.float32))
        svc.upsert_chunks([good, bad])

        call_data = client.upsert.call_args.kwargs["data"]
        assert len(call_data) == 1
        assert call_data[0]["chunk_id"] == good.chunk_id

    def test_upsert_empty_list_returns_zero(self):
        from storage.milvus.milvus_service import MilvusService

        client = self._make_mock_client()
        svc = MilvusService(client=client)
        assert svc.upsert_chunks([]) == 0
        client.upsert.assert_not_called()

    def test_search_raises_on_wrong_dim(self):
        from storage.milvus.milvus_service import MilvusService

        client = self._make_mock_client()
        svc = MilvusService(client=client)
        with pytest.raises(ValueError, match="dim"):
            svc.search(np.zeros(128, dtype=np.float32))

    def test_delete_validates_empty_media_id(self):
        from storage.milvus.milvus_service import MilvusService

        client = self._make_mock_client()
        svc = MilvusService(client=client)
        with pytest.raises(ValueError):
            svc.delete_by_media_id("")
