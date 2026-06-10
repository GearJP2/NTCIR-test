from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GroundTruthMoment:
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    media_id: str
    query: str
    ground_truth: GroundTruthMoment


@dataclass(frozen=True)
class EvaluationVideo:
    media_id: str
    video_path: Path
    duration_sec: float | None
    queries: tuple[EvaluationQuery, ...]


def load_evaluation_manifest(path: Path) -> list[EvaluationVideo]:
    """
    Load an ActivityNet-style evaluation manifest.

    The manifest is JSONL. Each line represents one source video:

    {
      "media_id": "v_123",
      "video_path": "data/activitynet/videos/v_123.mp4",
      "duration_sec": 120.4,
      "queries": [
        {
          "query_id": "v_123:0",
          "query": "A woman is doing sit ups",
          "ground_truth": {"start_sec": 39.8, "end_sec": 54.6}
        }
      ]
    }
    """
    videos: list[EvaluationVideo] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                videos.append(_parse_video(json.loads(line), base_dir=path.parent))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid manifest row at {path}:{line_no}: {exc}") from exc
    return videos


def iter_evaluation_queries(videos: list[EvaluationVideo]) -> list[EvaluationQuery]:
    return [query for video in videos for query in video.queries]


def _parse_video(row: dict[str, Any], base_dir: Path) -> EvaluationVideo:
    media_id = str(row["media_id"])
    video_path = _resolve_video_path(Path(str(row["video_path"])), base_dir)

    queries = tuple(_parse_query(media_id, item, i) for i, item in enumerate(row["queries"]))
    if not queries:
        raise ValueError(f"media_id={media_id} has no evaluation queries")

    duration = row.get("duration_sec")
    return EvaluationVideo(
        media_id=media_id,
        video_path=video_path,
        duration_sec=float(duration) if duration is not None else None,
        queries=queries,
    )


def _resolve_video_path(video_path: Path, base_dir: Path) -> Path:
    if video_path.is_absolute():
        return video_path

    base_relative = base_dir / video_path
    if base_relative.exists() or not video_path.exists():
        return base_relative
    return video_path


def _parse_query(media_id: str, row: dict[str, Any], index: int) -> EvaluationQuery:
    gt = row["ground_truth"]
    start_sec = float(gt["start_sec"])
    end_sec = float(gt["end_sec"])
    if end_sec <= start_sec:
        raise ValueError(
            f"media_id={media_id} query_index={index} has non-positive ground-truth duration"
        )

    return EvaluationQuery(
        query_id=str(row.get("query_id") or f"{media_id}:{index}"),
        media_id=media_id,
        query=str(row["query"]),
        ground_truth=GroundTruthMoment(start_sec=start_sec, end_sec=end_sec),
    )
