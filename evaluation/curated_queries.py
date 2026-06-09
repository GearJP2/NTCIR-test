from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CuratedQuery:
    query_id: str
    query: str
    tags: tuple[str, ...] = ()
    notes: str = ""
    media_id: str | None = None


def load_curated_queries(path: Path) -> list[CuratedQuery]:
    queries: list[CuratedQuery] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                queries.append(_parse_query(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid curated query at {path}:{line_no}: {exc}") from exc
    return queries


def _parse_query(row: dict[str, Any]) -> CuratedQuery:
    query_id = str(row["query_id"]).strip()
    query = str(row["query"]).strip()
    if not query_id:
        raise ValueError("query_id is required")
    if len(query) < 3:
        raise ValueError("query must be at least 3 characters")

    raw_tags = row.get("tags", [])
    if not isinstance(raw_tags, list):
        raise ValueError("tags must be a list")
    tags = tuple(str(tag).strip() for tag in raw_tags if str(tag).strip())

    notes = str(row.get("notes", "")).strip()
    media_id = row.get("media_id")
    return CuratedQuery(
        query_id=query_id,
        query=query,
        tags=tags,
        notes=notes,
        media_id=str(media_id).strip() if media_id else None,
    )
