import numpy as np

from services.events.visual_boundaries import (
    BoundaryPoint,
    VisualSample,
    adjacent_cosine_distances,
    build_event_intervals,
    pool_event_embedding,
    select_boundary_points,
)


def sample(timestamp_ms: int, values: list[float]) -> VisualSample:
    return VisualSample(timestamp_ms, np.array(values, dtype=np.float32))


def test_adjacent_cosine_distances_detects_visual_change():
    samples = [
        sample(0, [1, 0]),
        sample(5_000, [1, 0]),
        sample(10_000, [0, 1]),
    ]

    assert np.allclose(adjacent_cosine_distances(samples), [0, 1])


def test_select_boundary_points_applies_peak_and_gap_rules():
    samples = [sample(i * 5_000, [1, 0]) for i in range(6)]
    scores = np.array([0.1, 0.8, 0.7, 0.9, 0.1], dtype=np.float32)

    boundaries = select_boundary_points(
        samples,
        scores,
        percentile=50,
        min_gap_ms=10_000,
        min_score=0.1,
    )

    assert [boundary.timestamp_ms for boundary in boundaries] == [10_000, 20_000]


def test_select_boundary_points_rejects_zero_score_plateaus():
    samples = [sample(i * 5_000, [1, 0]) for i in range(6)]

    assert select_boundary_points(samples, np.zeros(5), min_score=0.01) == []


def test_build_event_intervals_splits_long_events_and_skips_tiny_boundaries():
    intervals = build_event_intervals(
        start_ms=0,
        end_ms=130_000,
        boundaries=[
            BoundaryPoint(5_000, 0.9),
            BoundaryPoint(70_000, 0.8),
        ],
        min_event_ms=10_000,
        max_event_ms=60_000,
    )

    assert [(event.start_ms, event.end_ms) for event in intervals] == [
        (0, 60_000),
        (60_000, 70_000),
        (70_000, 130_000),
    ]


def test_pool_event_embedding_normalizes_mean_visual_embedding():
    pooled = pool_event_embedding(
        [sample(0, [1, 0]), sample(5_000, [0, 1])],
        build_event_intervals(start_ms=0, end_ms=10_000, boundaries=[])[0],
    )

    assert np.allclose(pooled, [2**-0.5, 2**-0.5])


def test_build_event_intervals_merges_short_trailing_remainder():
    intervals = build_event_intervals(
        start_ms=0,
        end_ms=65_000,
        boundaries=[],
        min_event_ms=10_000,
        max_event_ms=60_000,
    )

    assert [(event.start_ms, event.end_ms) for event in intervals] == [
        (0, 65_000)
    ]
