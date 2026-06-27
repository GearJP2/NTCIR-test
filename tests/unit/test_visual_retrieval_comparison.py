import numpy as np

from evaluation.visual_retrieval import (
    VisualCandidate,
    recall_at_k,
    rank_visual_candidates,
    reciprocal_rank_fuse_visual_candidates,
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


def test_recall_at_k_counts_hits_at_or_above_cutoff():
    assert recall_at_k([1, 3, 11, None], 3) == 0.5


def test_reciprocal_rank_fusion_can_weight_candidate_lists():
    semantic = VisualCandidate("semantic-a", 0, 10_000, np.array([1.0, 0.0]))
    fixed = VisualCandidate("fixed-a", 0, 120_000, np.array([0.0, 1.0]))

    fused = reciprocal_rank_fuse_visual_candidates(
        [[(semantic, 0.1)], [(fixed, 0.9)]],
        weights=[1.0, 0.5],
    )

    assert [candidate.candidate_id for candidate, _score in fused] == [
        "semantic-a",
        "fixed-a",
    ]
