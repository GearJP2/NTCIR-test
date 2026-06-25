import numpy as np

from services.dataset.castle_transcripts import TranscriptSpan
from services.events.transcript_boundaries import (
    contextual_transcript_distances,
    fuse_boundary_scores,
    transcript_text_bins,
)


def test_transcript_text_bins_attach_overlapping_spans():
    spans = [
        TranscriptSpan("s1", "v1", 0, 6_000, "first", 0),
        TranscriptSpan("s2", "v1", 6_000, 10_000, "second", 1),
    ]

    assert transcript_text_bins(
        [0, 5_000],
        end_ms=10_000,
        spans=spans,
    ) == ["first", "first second"]


def test_contextual_transcript_distances_marks_missing_context_unavailable():
    scores = contextual_transcript_distances(
        np.array([[1, 0], [0, 1], [0, 1], [0, 1]], dtype=np.float32),
        np.array([True, False, False, True]),
        context_radius=1,
    )

    assert scores.available.tolist() == [False, True, False]
    assert scores.scores[0] == 0.0
    assert scores.scores[1] > 0.0


def test_fuse_boundary_scores_preserves_visual_score_without_transcript():
    fused = fuse_boundary_scores(
        np.array([0.2, 0.4], dtype=np.float32),
        np.array([0.8, 1.0], dtype=np.float32),
        np.array([True, False]),
        transcript_weight=0.25,
    )

    assert np.allclose(fused, [0.4, 0.4])
