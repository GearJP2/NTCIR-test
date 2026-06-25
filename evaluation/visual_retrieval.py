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


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)
