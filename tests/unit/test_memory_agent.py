import numpy as np
import pytest

from services.memory.memory_store import MemoryStore
from services.memory.temporal_index import TemporalIndex


def test_memory_store_add_and_retrieve():
    store = MemoryStore()
    vec = np.random.rand(512).astype(np.float32)
    store.add_node("seg1", "media1", 0.0, 5.0, "summary text", vec)
    assert store.has_node("seg1")
    node = store.get_node("seg1")
    assert node["media_id"] == "media1"
    assert np.allclose(node["embedding"], vec)


def test_temporal_index_neighbours():
    idx = TemporalIndex()
    idx.add_segment("s1", prev_segment_id=None)
    idx.add_segment("s2", prev_segment_id="s1")
    idx.add_segment("s3", prev_segment_id="s2")

    neighbours = idx.get_neighbours("s2", hops=1)
    assert "s1" in neighbours
    assert "s3" in neighbours
    assert "s2" not in neighbours


def test_temporal_index_no_neighbours_unknown_id():
    idx = TemporalIndex()
    assert idx.get_neighbours("nonexistent") == []
