from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VisualSample:
    timestamp_ms: int
    embedding: np.ndarray


@dataclass(frozen=True)
class BoundaryPoint:
    timestamp_ms: int
    score: float


@dataclass(frozen=True)
class EventInterval:
    start_ms: int
    end_ms: int
    boundary_confidence: float


def adjacent_cosine_distances(samples: list[VisualSample]) -> np.ndarray:
    if len(samples) < 2:
        return np.array([], dtype=np.float32)
    embeddings = np.stack([sample.embedding for sample in samples]).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return np.clip(1.0 - similarities, 0.0, 2.0).astype(np.float32)


def smooth_scores(scores: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0 or len(scores) == 0:
        return scores.astype(np.float32)
    kernel = np.ones(radius * 2 + 1, dtype=np.float32)
    kernel /= kernel.sum()
    return np.convolve(scores, kernel, mode="same").astype(np.float32)


def select_boundary_points(
    samples: list[VisualSample],
    scores: np.ndarray,
    *,
    percentile: float = 85.0,
    min_gap_ms: int = 10_000,
    min_score: float = 0.01,
) -> list[BoundaryPoint]:
    if len(scores) != max(0, len(samples) - 1):
        raise ValueError("scores must describe each adjacent sample pair")
    if len(scores) == 0:
        return []

    threshold = float(np.percentile(scores, percentile))
    candidates = [
        BoundaryPoint(
            timestamp_ms=samples[index + 1].timestamp_ms,
            score=float(score),
        )
        for index, score in enumerate(scores)
        if score >= threshold
        and score >= min_score
        and (index == 0 or score >= scores[index - 1])
        and (index == len(scores) - 1 or score >= scores[index + 1])
    ]
    selected: list[BoundaryPoint] = []
    for candidate in sorted(candidates, key=lambda point: point.score, reverse=True):
        if all(
            abs(candidate.timestamp_ms - existing.timestamp_ms) >= min_gap_ms
            for existing in selected
        ):
            selected.append(candidate)
    return sorted(selected, key=lambda point: point.timestamp_ms)


def build_event_intervals(
    *,
    start_ms: int,
    end_ms: int,
    boundaries: list[BoundaryPoint],
    min_event_ms: int = 10_000,
    max_event_ms: int = 60_000,
) -> list[EventInterval]:
    if end_ms <= start_ms:
        raise ValueError("end_ms must be greater than start_ms")
    valid_boundaries = [
        boundary
        for boundary in boundaries
        if start_ms < boundary.timestamp_ms < end_ms
    ]
    intervals: list[EventInterval] = []
    cursor = start_ms
    pending_confidence = 1.0

    for boundary in [*valid_boundaries, BoundaryPoint(end_ms, 1.0)]:
        proposed_end = boundary.timestamp_ms
        while proposed_end - cursor > max_event_ms:
            split_end = cursor + max_event_ms
            intervals.append(EventInterval(cursor, split_end, 0.0))
            cursor = split_end
            pending_confidence = 0.0
        if proposed_end - cursor < min_event_ms and proposed_end != end_ms:
            continue
        if proposed_end > cursor:
            confidence = (
                boundary.score if proposed_end != end_ms else pending_confidence
            )
            intervals.append(EventInterval(cursor, proposed_end, confidence))
            cursor = proposed_end
            pending_confidence = boundary.score

    if intervals and intervals[-1].end_ms < end_ms:
        last = intervals[-1]
        intervals[-1] = EventInterval(
            start_ms=last.start_ms,
            end_ms=end_ms,
            boundary_confidence=last.boundary_confidence,
        )
    if (
        len(intervals) > 1
        and intervals[-1].end_ms - intervals[-1].start_ms < min_event_ms
    ):
        previous = intervals[-2]
        trailing = intervals[-1]
        intervals[-2:] = [
            EventInterval(
                start_ms=previous.start_ms,
                end_ms=trailing.end_ms,
                boundary_confidence=previous.boundary_confidence,
            )
        ]
    return intervals


def pool_event_embedding(
    samples: list[VisualSample],
    interval: EventInterval,
) -> np.ndarray:
    selected = [
        sample.embedding
        for sample in samples
        if interval.start_ms <= sample.timestamp_ms < interval.end_ms
    ]
    if not selected:
        raise ValueError("event interval contains no visual samples")
    pooled = np.mean(np.stack(selected), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(pooled))
    return pooled / max(norm, 1e-12)
