import numpy as np
import pytest

from services.embedding.fusion import fuse_embeddings


def test_fuse_all_modalities():
    a = np.ones(512, dtype=np.float32)
    v = np.ones(768, dtype=np.float32)
    t = np.ones(768, dtype=np.float32)
    result = fuse_embeddings(a, v, t)
    assert result.dtype == np.float32
    assert abs(np.linalg.norm(result) - 1.0) < 1e-5


def test_fuse_text_only():
    t = np.random.rand(768).astype(np.float32)
    result = fuse_embeddings(None, None, t)
    assert abs(np.linalg.norm(result) - 1.0) < 1e-5


def test_fuse_raises_on_no_modality():
    with pytest.raises(ValueError):
        fuse_embeddings(None, None, None)
