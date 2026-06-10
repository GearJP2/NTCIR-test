from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from storage.milvus.schemas import AUDIO_COLLECTION, TEXT_COLLECTION, VISUAL_COLLECTION


@dataclass(frozen=True)
class CollectionReadiness:
    collection_name: str
    source_type: str
    sample_count: int
    sample_limit: int
    ready: bool
    error: str | None = None


def inspect_media_index(
    media_id: str,
    milvus_client,
    sample_limit: int = 100,
) -> dict[str, Any]:
    if sample_limit <= 0:
        raise ValueError("sample_limit must be positive")

    collections = [
        (VISUAL_COLLECTION, "visual", ["frame_id", "media_id", "timestamp_sec"]),
        (TEXT_COLLECTION, "asr", ["chunk_id", "media_id", "start_sec", "end_sec", "text"]),
        (AUDIO_COLLECTION, "audio", ["segment_id", "media_id", "start_sec", "end_sec"]),
    ]
    readiness = [
        _inspect_collection(
            milvus_client=milvus_client,
            collection_name=collection_name,
            source_type=source_type,
            output_fields=output_fields,
            media_id=media_id,
            sample_limit=sample_limit,
        )
        for collection_name, source_type, output_fields in collections
    ]

    return {
        "media_id": media_id,
        "sample_limit": sample_limit,
        "ready": any(item.ready for item in readiness),
        "collections": [item.__dict__ for item in readiness],
    }


def list_indexed_media_candidates(
    milvus_client,
    sample_limit: int = 1000,
) -> list[dict[str, Any]]:
    if sample_limit <= 0:
        raise ValueError("sample_limit must be positive")

    collection_specs = [
        (VISUAL_COLLECTION, "visual", ["media_id"]),
        (TEXT_COLLECTION, "asr", ["media_id"]),
        (AUDIO_COLLECTION, "audio", ["media_id"]),
    ]
    counts_by_media: dict[str, dict[str, int]] = {}

    for collection_name, source_type, output_fields in collection_specs:
        try:
            rows = milvus_client.query(
                collection_name=collection_name,
                filter='media_id != ""',
                output_fields=output_fields,
                limit=sample_limit,
            )
        except Exception:
            rows = []

        for row in rows or []:
            media_id = row.get("media_id")
            if not media_id:
                continue
            media_counts = counts_by_media.setdefault(
                str(media_id),
                {"visual": 0, "asr": 0, "audio": 0},
            )
            media_counts[source_type] += 1

    candidates = [
        {
            "media_id": media_id,
            "visual": counts["visual"],
            "asr": counts["asr"],
            "audio": counts["audio"],
            "total": counts["visual"] + counts["asr"] + counts["audio"],
        }
        for media_id, counts in counts_by_media.items()
    ]
    return sorted(candidates, key=lambda item: item["total"], reverse=True)


def diagnostics_to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)


def _inspect_collection(
    milvus_client,
    collection_name: str,
    source_type: str,
    output_fields: list[str],
    media_id: str,
    sample_limit: int,
) -> CollectionReadiness:
    try:
        rows = milvus_client.query(
            collection_name=collection_name,
            filter=f'media_id == "{media_id}"',
            output_fields=output_fields,
            limit=sample_limit,
        )
    except Exception as exc:
        return CollectionReadiness(
            collection_name=collection_name,
            source_type=source_type,
            sample_count=0,
            sample_limit=sample_limit,
            ready=False,
            error=str(exc),
        )

    count = len(rows or [])
    return CollectionReadiness(
        collection_name=collection_name,
        source_type=source_type,
        sample_count=count,
        sample_limit=sample_limit,
        ready=count > 0,
        error=None,
    )
