from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from services.events.transcript_boundaries import fuse_boundary_scores
from services.events.visual_boundaries import (
    BoundaryPoint,
    EventInterval,
    VisualSample,
    adjacent_cosine_distances,
    build_event_intervals,
    select_boundary_points,
    smooth_scores,
)

VisualDetector = Literal["v1", "v2"]


@dataclass(frozen=True)
class VisualBoundaryScores:
    detector: VisualDetector
    scores: np.ndarray


@dataclass(frozen=True)
class VisualSegmentationConfig:
    name: str
    detector: VisualDetector
    context_radius: int = 2
    smoothing_radius: int = 1
    boundary_percentile: float = 85.0
    min_boundary_score: float = 0.01
    min_event_ms: int = 10_000
    max_event_ms: int = 60_000


@dataclass(frozen=True)
class VisualSegmentationSummary:
    name: str
    detector: VisualDetector
    learned_boundary_count: int
    forced_split_count: int
    event_count: int
    min_duration_ms: int
    max_duration_ms: int
    mean_duration_ms: float


@dataclass(frozen=True)
class VisualSegmentationResult:
    config: VisualSegmentationConfig
    raw_scores: np.ndarray
    visual_scores: np.ndarray
    transcript_scores: np.ndarray | None
    transcript_available: np.ndarray | None
    scores: np.ndarray
    boundaries: list[BoundaryPoint]
    intervals: list[EventInterval]
    summary: VisualSegmentationSummary


def score_visual_boundaries(
    samples: list[VisualSample],
    *,
    detector: VisualDetector,
    context_radius: int = 2,
) -> VisualBoundaryScores:
    if detector == "v1":
        return VisualBoundaryScores(
            detector=detector,
            scores=adjacent_cosine_distances(samples),
        )
    if detector == "v2":
        return VisualBoundaryScores(
            detector=detector,
            scores=_contextual_cosine_distances(samples, context_radius),
        )
    raise ValueError(f"unknown visual detector: {detector}")


def run_visual_segmentation(
    samples: list[VisualSample],
    config: VisualSegmentationConfig,
    *,
    start_ms: int,
    end_ms: int,
    transcript_scores: np.ndarray | None = None,
    transcript_available: np.ndarray | None = None,
    transcript_weight: float = 0.0,
) -> VisualSegmentationResult:
    visual_scores = score_visual_boundaries(
        samples,
        detector=config.detector,
        context_radius=config.context_radius,
    ).scores
    if transcript_scores is None:
        raw_scores = visual_scores
    else:
        if transcript_available is None:
            raise ValueError(
                "transcript_available is required with transcript_scores"
            )
        raw_scores = fuse_boundary_scores(
            visual_scores,
            transcript_scores,
            transcript_available,
            transcript_weight=transcript_weight,
        )
    scores = smooth_scores(raw_scores, radius=config.smoothing_radius)
    boundaries = select_boundary_points(
        samples,
        scores,
        percentile=config.boundary_percentile,
        min_gap_ms=config.min_event_ms,
        min_score=config.min_boundary_score,
    )
    intervals = build_event_intervals(
        start_ms=start_ms,
        end_ms=end_ms,
        boundaries=boundaries,
        min_event_ms=config.min_event_ms,
        max_event_ms=config.max_event_ms,
    )
    durations = [interval.end_ms - interval.start_ms for interval in intervals]
    summary = VisualSegmentationSummary(
        name=config.name,
        detector=config.detector,
        learned_boundary_count=len(boundaries),
        forced_split_count=sum(
            interval.boundary_confidence == 0.0 for interval in intervals
        ),
        event_count=len(intervals),
        min_duration_ms=min(durations),
        max_duration_ms=max(durations),
        mean_duration_ms=float(np.mean(durations)),
    )
    return VisualSegmentationResult(
        config=config,
        raw_scores=raw_scores,
        visual_scores=visual_scores,
        transcript_scores=transcript_scores,
        transcript_available=transcript_available,
        scores=scores,
        boundaries=boundaries,
        intervals=intervals,
        summary=summary,
    )


def _contextual_cosine_distances(
    samples: list[VisualSample],
    context_radius: int,
) -> np.ndarray:
    if context_radius < 1:
        raise ValueError("context_radius must be at least 1")
    if len(samples) < 2:
        return np.array([], dtype=np.float32)

    embeddings = np.stack([sample.embedding for sample in samples]).astype(np.float32)
    scores: list[float] = []
    for boundary_index in range(len(samples) - 1):
        left_start = max(0, boundary_index - context_radius)
        right_end = min(len(samples), boundary_index + context_radius + 2)
        left = _normalize(np.mean(embeddings[left_start : boundary_index + 1], axis=0))
        right = _normalize(
            np.mean(embeddings[boundary_index + 1 : right_end], axis=0)
        )
        scores.append(max(0.0, 1.0 - float(np.dot(left, right))))
    return np.array(scores, dtype=np.float32)


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)
