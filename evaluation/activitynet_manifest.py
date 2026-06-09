from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov")


def build_activitynet_manifest_rows(
    dataset_rows: Iterable[dict[str, Any]],
    video_root: Path,
    max_videos: int = 50,
    max_queries: int | None = 500,
    seed: int = 19,
    require_video_file: bool = True,
) -> list[dict[str, Any]]:
    candidates = []
    for row in dataset_rows:
        parsed = _parse_activitynet_row(row, video_root)
        if parsed is None:
            continue
        if require_video_file and not Path(parsed["video_path"]).exists():
            continue
        candidates.append(parsed)

    rng = random.Random(seed)
    rng.shuffle(candidates)

    selected: list[dict[str, Any]] = []
    query_count = 0
    for candidate in candidates:
        if len(selected) >= max_videos:
            break
        if max_queries is not None and query_count >= max_queries:
            break

        queries = candidate["queries"]
        if max_queries is not None:
            remaining = max_queries - query_count
            queries = queries[:remaining]
        if not queries:
            continue

        selected.append({**candidate, "queries": queries})
        query_count += len(queries)

    return selected


def write_manifest_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_activitynet_row(row: dict[str, Any], video_root: Path) -> dict[str, Any] | None:
    media_id = _first_text(row, ("video_id", "media_id", "id", "youtube_id", "vid"))
    if not media_id:
        return None

    sentences = row.get("sentences") or row.get("captions") or row.get("caption")
    timestamps = row.get("timestamps") or row.get("segments")
    if isinstance(sentences, str):
        sentences = [sentences]
    if not isinstance(sentences, list) or not isinstance(timestamps, list):
        return None

    video_path = _resolve_video_path(row, video_root, media_id)
    queries = []
    for index, (sentence, timestamp) in enumerate(zip(sentences, timestamps, strict=False)):
        if not isinstance(timestamp, (list, tuple)) or len(timestamp) < 2:
            continue
        start_sec = float(timestamp[0])
        end_sec = float(timestamp[1])
        if end_sec <= start_sec:
            continue
        query = str(sentence).strip()
        if not query:
            continue
        queries.append(
            {
                "query_id": f"{media_id}:{index}",
                "query": query,
                "ground_truth": {
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                },
            }
        )

    if not queries:
        return None

    duration = row.get("duration") or row.get("duration_sec")
    return {
        "media_id": media_id,
        "video_path": str(video_path),
        "duration_sec": float(duration) if duration is not None else None,
        "queries": queries,
    }


def _resolve_video_path(row: dict[str, Any], video_root: Path, media_id: str) -> Path:
    explicit = _first_text(row, ("video_path", "path", "file", "filename"))
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else video_root / path

    for ext in VIDEO_EXTENSIONS:
        candidate = video_root / f"{media_id}{ext}"
        if candidate.exists():
            return candidate

    return video_root / f"{media_id}.mp4"


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None
