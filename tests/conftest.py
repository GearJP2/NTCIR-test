import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── App fixture ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def client():
    from app.main import app
    return TestClient(app)


# ── Mock Milvus ───────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_milvus(monkeypatch):
    mock = MagicMock()
    mock.list_collections.return_value = []
    mock.has_collection.return_value = True
    mock.search.return_value = [[]]
    monkeypatch.setattr("storage.milvus.client.get_milvus_client", lambda: mock)
    return mock


# ── Mock MinIO ────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_minio(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("storage.minio.client.get_minio_client", lambda: mock)
    return mock


# ── Dummy audio file ──────────────────────────────────────────────────────────
@pytest.fixture
def sample_wav(tmp_path) -> Path:
    import soundfile as sf
    path = tmp_path / "sample.wav"
    sf.write(str(path), np.zeros(16000, dtype=np.float32), 16000)
    return path
