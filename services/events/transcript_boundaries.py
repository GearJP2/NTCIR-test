from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from services.dataset.castle_transcripts import TranscriptSpan


@dataclass(frozen=True)
class TranscriptBoundaryScores:
    scores: np.ndarray
    available: np.ndarray


def transcript_text_bins(
    timestamps_ms: list[int],
    *,
    end_ms: int,
    spans: list[TranscriptSpan],
) -> list[str]:
    if not timestamps_ms:
        return []
    interval_ends = [*timestamps_ms[1:], end_ms]
    return [
        " ".join(
            span.text
            for span in spans
            if span.start_ms < interval_end and span.end_ms > interval_start
        ).strip()
        for interval_start, interval_end in zip(
            timestamps_ms,
            interval_ends,
            strict=True,
        )
    ]


def contextual_transcript_distances(
    embeddings: np.ndarray,
    available: np.ndarray,
    *,
    context_radius: int,
) -> TranscriptBoundaryScores:
    if context_radius < 1:
        raise ValueError("context_radius must be at least 1")
    embeddings = np.asarray(embeddings, dtype=np.float32)
    available = np.asarray(available, dtype=bool)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if len(embeddings) != len(available):
        raise ValueError("availability must describe each transcript embedding")
    if len(embeddings) < 2:
        return TranscriptBoundaryScores(
            scores=np.array([], dtype=np.float32),
            available=np.array([], dtype=bool),
        )

    scores: list[float] = []
    boundary_available: list[bool] = []
    for boundary_index in range(len(embeddings) - 1):
        left_start = max(0, boundary_index - context_radius)
        right_end = min(len(embeddings), boundary_index + context_radius + 2)
        left = embeddings[left_start : boundary_index + 1][
            available[left_start : boundary_index + 1]
        ]
        right = embeddings[boundary_index + 1 : right_end][
            available[boundary_index + 1 : right_end]
        ]
        has_context = len(left) > 0 and len(right) > 0
        boundary_available.append(has_context)
        if not has_context:
            scores.append(0.0)
            continue
        left_pooled = _normalize(np.mean(left, axis=0))
        right_pooled = _normalize(np.mean(right, axis=0))
        scores.append(max(0.0, 1.0 - float(np.dot(left_pooled, right_pooled))))

    return TranscriptBoundaryScores(
        scores=np.asarray(scores, dtype=np.float32),
        available=np.asarray(boundary_available, dtype=bool),
    )


def fuse_boundary_scores(
    visual_scores: np.ndarray,
    transcript_scores: np.ndarray,
    transcript_available: np.ndarray,
    *,
    transcript_weight: float,
) -> np.ndarray:
    if not 0.0 <= transcript_weight <= 1.0:
        raise ValueError("transcript_weight must be between 0 and 1")
    visual_scores = np.asarray(visual_scores, dtype=np.float32)
    transcript_scores = np.asarray(transcript_scores, dtype=np.float32)
    transcript_available = np.asarray(transcript_available, dtype=bool)
    if not (
        visual_scores.shape
        == transcript_scores.shape
        == transcript_available.shape
    ):
        raise ValueError("visual, transcript, and availability shapes must match")

    fused = visual_scores.copy()
    fused[transcript_available] = (
        visual_scores[transcript_available]
        + transcript_weight * transcript_scores[transcript_available]
    )
    return fused


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)
