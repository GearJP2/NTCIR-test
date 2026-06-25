import numpy as np

from services.events.visual_boundaries import VisualSample
from services.events.visual_segmentation import (
    VisualSegmentationConfig,
    run_visual_segmentation,
    score_visual_boundaries,
)


def _sample(timestamp_ms: int, values: list[float]) -> VisualSample:
    vector = np.array(values, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return VisualSample(timestamp_ms=timestamp_ms, embedding=vector)


def test_v2_suppresses_transient_outlier_and_retains_sustained_transition():
    samples = [
        _sample(0, [1.0, 0.0]),
        _sample(5_000, [1.0, 0.0]),
        _sample(10_000, [0.0, 1.0]),  # transient head turn / blur
        _sample(15_000, [1.0, 0.0]),
        _sample(20_000, [0.0, 1.0]),  # sustained scene transition begins
        _sample(25_000, [0.0, 1.0]),
        _sample(30_000, [0.0, 1.0]),
    ]

    result = score_visual_boundaries(samples, detector="v2", context_radius=2)

    assert result.scores[2] < 0.25
    assert result.scores[3] > 0.5


def test_segmentation_run_reports_learned_boundaries_and_forced_splits():
    samples = [
        _sample(index * 5_000, [1.0, 0.0] if index < 14 else [0.0, 1.0])
        for index in range(27)
    ]

    result = run_visual_segmentation(
        samples,
        VisualSegmentationConfig(
            name="v2-test",
            detector="v2",
            context_radius=2,
            boundary_percentile=80,
            min_boundary_score=0.1,
            min_event_ms=10_000,
            max_event_ms=60_000,
        ),
        start_ms=0,
        end_ms=130_000,
    )

    assert result.summary.learned_boundary_count == 1
    assert result.summary.forced_split_count == 1
    assert result.summary.event_count == 3
    assert result.summary.min_duration_ms == 10_000
    assert result.summary.max_duration_ms == 60_000
