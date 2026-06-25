import numpy as np

from scripts.build_visual_semantic_events import _median_step_ms, _normalize


def test_median_step_ms_handles_regular_samples():
    assert _median_step_ms([0, 5_000, 10_000, 15_000]) == 5_000


def test_normalize_returns_unit_vector():
    assert np.allclose(_normalize(np.array([3, 4])), [0.6, 0.8])
