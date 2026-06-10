from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.schemas.search import Evidence, VideoMoment

SourceType = Literal["visual", "audio", "asr", "summary"]


@dataclass(frozen=True)
class VideoWindow:
    media_id: str
    start_sec: float
    end_sec: float

    @property
    def moment_id(self) -> str:
        return format_moment_id(self.media_id, self.start_sec, self.end_sec)

    @property
    def thumbnail_sec(self) -> float:
        return (self.start_sec + self.end_sec) / 2.0


@dataclass(frozen=True)
class EvidenceHit:
    source_type: SourceType
    media_id: str
    score: float
    source_id: str | None = None
    timestamp_sec: float | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    text: str | None = None


def generate_fixed_windows(
    media_id: str,
    duration_sec: float,
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
) -> list[VideoWindow]:
    if duration_sec <= 0.0:
        return []
    if window_sec <= 0.0:
        raise ValueError("window_sec must be positive")
    if stride_sec <= 0.0:
        raise ValueError("stride_sec must be positive")

    if duration_sec <= window_sec:
        return [VideoWindow(media_id=media_id, start_sec=0.0, end_sec=float(duration_sec))]

    windows: list[VideoWindow] = []
    start = 0.0
    while start + window_sec <= duration_sec:
        windows.append(
            VideoWindow(
                media_id=media_id,
                start_sec=round(start, 6),
                end_sec=round(start + window_sec, 6),
            )
        )
        start += stride_sec

    last_end = windows[-1].end_sec if windows else 0.0
    if last_end < duration_sec:
        tail_start = max(0.0, duration_sec - window_sec)
        if not windows or abs(tail_start - windows[-1].start_sec) > 1e-6:
            windows.append(
                VideoWindow(
                    media_id=media_id,
                    start_sec=round(tail_start, 6),
                    end_sec=round(duration_sec, 6),
                )
            )

    return windows


def evidence_hits_to_video_moments(
    media_id: str,
    windows: list[VideoWindow],
    hits: list[EvidenceHit],
    top_k: int = 10,
    source_weights: dict[SourceType, float] | None = None,
) -> list[VideoMoment]:
    weights = source_weights or {
        "visual": 1.0,
        "audio": 1.0,
        "asr": 1.0,
        "summary": 1.0,
    }

    scored: list[tuple[float, float, VideoWindow, list[Evidence]]] = []
    for window in windows:
        matched_hits = [
            hit for hit in hits if hit.media_id == media_id and _hit_overlaps_window(hit, window)
        ]
        if not matched_hits:
            continue

        evidence = [_to_evidence(hit) for hit in matched_hits]
        raw_score = max(hit.score * weights.get(hit.source_type, 1.0) for hit in matched_hits)
        scored.append((raw_score, _public_score(raw_score), window, evidence))

    ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
    return [
        VideoMoment(
            rank=rank,
            moment_id=window.moment_id,
            media_id=window.media_id,
            start_sec=window.start_sec,
            end_sec=window.end_sec,
            score=score,
            thumbnail_sec=window.thumbnail_sec,
            evidence=evidence,
        )
        for rank, (_raw_score, score, window, evidence) in enumerate(ranked, start=1)
    ]


def format_moment_id(media_id: str, start_sec: float, end_sec: float) -> str:
    return f"{media_id}:{start_sec:.3f}-{end_sec:.3f}"


def _hit_overlaps_window(hit: EvidenceHit, window: VideoWindow) -> bool:
    if hit.timestamp_sec is not None:
        return window.start_sec <= hit.timestamp_sec < window.end_sec

    if hit.start_sec is None or hit.end_sec is None:
        return False

    return min(hit.end_sec, window.end_sec) > max(hit.start_sec, window.start_sec)


def _to_evidence(hit: EvidenceHit) -> Evidence:
    return Evidence(
        source_type=hit.source_type,
        score=_public_score(hit.score),
        source_id=hit.source_id,
        timestamp_sec=hit.timestamp_sec,
        start_sec=hit.start_sec,
        end_sec=hit.end_sec,
        text=hit.text,
    )


def _public_score(score: float) -> float:
    return max(0.0, score)
