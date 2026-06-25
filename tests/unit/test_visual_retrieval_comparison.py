import numpy as np

from evaluation.visual_retrieval import (
    VisualCandidate,
    rank_visual_candidates,
    temporal_iou,
    temporal_overlap_ratio,
)


def test_rank_visual_candidates_uses_direct_visual_similarity():
    candidates = [
        VisualCandidate("event-a", 0, 10_000, np.array([1.0, 0.0])),
        VisualCandidate("event-b", 10_000, 20_000, np.array([0.0, 1.0])),
    ]

    ranked = rank_visual_candidates(np.array([0.9, 0.1]), candidates)

    assert [candidate.candidate_id for candidate, _ in ranked] == [
        "event-a",
        "event-b",
    ]


def test_temporal_overlap_ratio_measures_query_interval_coverage():
    assert temporal_overlap_ratio(5_000, 15_000, 10_000, 20_000) == 0.5


def test_temporal_iou_penalizes_oversized_candidate():
    assert temporal_iou(0, 120_000, 40_000, 60_000) == 1 / 6
