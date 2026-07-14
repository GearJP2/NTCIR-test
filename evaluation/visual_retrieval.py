from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VisualCandidate:
    candidate_id: str
    start_ms: int
    end_ms: int
    embedding: np.ndarray


def rank_visual_candidates(
    query_embedding: np.ndarray,
    candidates: list[VisualCandidate],
) -> list[tuple[VisualCandidate, float]]:
    query = _normalize(query_embedding)
    scored = [
        (candidate, float(np.dot(query, _normalize(candidate.embedding))))
        for candidate in candidates
    ]
    return sorted(scored, key=lambda item: item[1], reverse=True)


def reciprocal_rank_fuse_visual_candidates(
    ranked_lists: list[list[tuple[VisualCandidate, float]]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[VisualCandidate, float]]:
    if k < 1:
        raise ValueError("k must be positive")
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError("weights must match ranked_lists length")

    scores: dict[str, float] = {}
    candidates: dict[str, VisualCandidate] = {}
    effective_weights = weights or [1.0] * len(ranked_lists)
    for ranked, weight in zip(ranked_lists, effective_weights, strict=True):
        for rank, (candidate, _score) in enumerate(ranked, start=1):
            candidates.setdefault(candidate.candidate_id, candidate)
            scores[candidate.candidate_id] = scores.get(
                candidate.candidate_id,
                0.0,
            ) + weight / (k + rank)

    return sorted(
        [
            (candidate, scores[candidate_id])
            for candidate_id, candidate in candidates.items()
        ],
        key=lambda item: item[1],
        reverse=True,
    )


def temporal_overlap_ratio(
    candidate_start_ms: int,
    candidate_end_ms: int,
    expected_start_ms: int,
    expected_end_ms: int,
) -> float:
    expected_duration = expected_end_ms - expected_start_ms
    if expected_duration <= 0:
        raise ValueError("expected interval must have positive duration")
    intersection = max(
        0,
        min(candidate_end_ms, expected_end_ms)
        - max(candidate_start_ms, expected_start_ms),
    )
    return intersection / expected_duration


def temporal_iou(
    candidate_start_ms: int,
    candidate_end_ms: int,
    expected_start_ms: int,
    expected_end_ms: int,
) -> float:
    intersection = max(
        0,
        min(candidate_end_ms, expected_end_ms)
        - max(candidate_start_ms, expected_start_ms),
    )
    union = (
        max(candidate_end_ms, expected_end_ms)
        - min(candidate_start_ms, expected_start_ms)
    )
    return intersection / union if union > 0 else 0.0


def temporal_precision(
    candidate_start_ms: int,
    candidate_end_ms: int,
    expected_start_ms: int,
    expected_end_ms: int,
) -> float:
    candidate_duration = candidate_end_ms - candidate_start_ms
    if candidate_duration <= 0:
        raise ValueError("candidate interval must have positive duration")
    intersection = max(
        0,
        min(candidate_end_ms, expected_end_ms)
        - max(candidate_start_ms, expected_start_ms),
    )
    return intersection / candidate_duration


def recall_at_k(hit_ranks: list[int | None], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    if not hit_ranks:
        return 0.0
    return sum(rank is not None and rank <= k for rank in hit_ranks) / len(
        hit_ranks
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)
