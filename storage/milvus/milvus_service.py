"""
MilvusService — manages the `csat_episodic_memory` collection.

This is the canonical store for all indexed audio chunks.
Each row represents one audio segment with its embedding vector and
all metadata needed to retrieve and play back the original audio.

Collection schema
-----------------
chunk_id        VARCHAR(64)   PK  — UUID of the audio chunk
media_id        VARCHAR(64)       — Parent media file ID
start_sec       FLOAT             — Start offset in the source file (seconds)
end_sec         FLOAT             — End offset
duration_sec    FLOAT             — Chunk length (seconds)
minio_url       VARCHAR(1024)     — Presigned MinIO URL (30-day TTL)
object_key      VARCHAR(512)      — MinIO object key (regenerate URL on expiry)
transcript      VARCHAR(2048)     — Optional ASR text for the chunk
language        VARCHAR(8)        — ISO language code (default "th")
embedding_model VARCHAR(64)       — Which model produced the vector
created_at      INT64             — Unix timestamp (seconds)
embedding       FLOAT_VECTOR(512) — L2-normalised embedding (CLAP=512)

Index: HNSW / COSINE  (switch to IVF_FLAT for very large collections)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from app.core.exceptions import StorageError
from storage.milvus.index_params import hnsw_cosine_index

logger = structlog.get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EPISODIC_COLLECTION = "csat_episodic_memory"

# CLAP produces 512-dim; set to 768 if switching to Wav2Vec2 as primary model
EMBEDDING_DIM = 512

_OUTPUT_FIELDS = [
    "chunk_id",
    "media_id",
    "start_sec",
    "end_sec",
    "duration_sec",
    "minio_url",
    "object_key",
    "transcript",
    "language",
    "embedding_model",
    "created_at",
]

_SEARCH_PARAMS = {
    "metric_type": "COSINE",
    "params": {"ef": 200},   # search-time depth; tune for recall/latency trade-off
}


# ── Service ───────────────────────────────────────────────────────────────────

class MilvusService:
    """
    High-level CRUD + search interface for `csat_episodic_memory`.

    The collection is created automatically on first use if it does not exist.

    Parameters
    ----------
    client:
        Inject a pre-built MilvusClient (useful in tests). When omitted, the
        application singleton from `storage.milvus.client` is used.
    embedding_dim:
        Vector dimension. Must match the dimension used at index creation.
        Defaults to 512 (CLAP). Change to 768 for Wav2Vec2.
    """

    def __init__(
        self,
        client: MilvusClient | None = None,
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        from storage.milvus.client import get_milvus_client

        self._client = client or get_milvus_client()
        self._dim = embedding_dim
        self._ensure_collection()

    # ── Collection lifecycle ─────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Idempotently create the collection + index if absent."""
        if self._client.has_collection(EPISODIC_COLLECTION):
            return

        schema = CollectionSchema(
            fields=[
                FieldSchema(
                    "chunk_id", DataType.VARCHAR,
                    max_length=64, is_primary=True,
                ),
                FieldSchema("media_id",        DataType.VARCHAR,       max_length=64),
                FieldSchema("start_sec",        DataType.FLOAT),
                FieldSchema("end_sec",          DataType.FLOAT),
                FieldSchema("duration_sec",     DataType.FLOAT),
                FieldSchema(
                    "minio_url", DataType.VARCHAR,
                    max_length=1024, default_value="",
                ),
                FieldSchema(
                    "object_key", DataType.VARCHAR,
                    max_length=512, default_value="",
                ),
                FieldSchema(
                    "transcript", DataType.VARCHAR,
                    max_length=2048, default_value="",
                ),
                FieldSchema(
                    "language", DataType.VARCHAR,
                    max_length=8, default_value="th",
                ),
                FieldSchema(
                    "embedding_model", DataType.VARCHAR,
                    max_length=64, default_value="clap",
                ),
                FieldSchema("created_at", DataType.INT64),
                FieldSchema(
                    "embedding", DataType.FLOAT_VECTOR, dim=self._dim
                ),
            ],
            description="NTCIR CSAT episodic memory — audio chunk semantic index",
            enable_dynamic_field=False,
        )

        try:
            self._client.create_collection(
                collection_name=EPISODIC_COLLECTION,
                schema=schema,
            )
            self._client.create_index(
                collection_name=EPISODIC_COLLECTION,
                index_params=hnsw_cosine_index("embedding", ef_construction=256),
            )
            self._client.load_collection(EPISODIC_COLLECTION)
            logger.info(
                "milvus_service.collection_created",
                collection=EPISODIC_COLLECTION,
                dim=self._dim,
            )
        except Exception as exc:
            raise StorageError(
                f"Failed to create Milvus collection '{EPISODIC_COLLECTION}': {exc}"
            ) from exc

    # ── Write operations ─────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list) -> int:
        """
        Upsert a list of `AudioChunk` objects into `csat_episodic_memory`.

        Chunks that have no embedding (empty array) are skipped with a warning
        so a partial embedding failure does not abort the entire batch.

        Returns
        -------
        int
            Number of rows actually upserted.

        Raises
        ------
        StorageError – Milvus write failure
        """
        if not chunks:
            return 0

        records: list[dict[str, Any]] = []
        skipped = 0

        for chunk in chunks:
            if chunk.embedding is None or chunk.embedding.size == 0:
                logger.warning(
                    "milvus_service.skip_no_embedding",
                    chunk_id=chunk.chunk_id,
                )
                skipped += 1
                continue

            if chunk.embedding.shape[0] != self._dim:
                logger.error(
                    "milvus_service.dim_mismatch",
                    chunk_id=chunk.chunk_id,
                    got=chunk.embedding.shape[0],
                    expected=self._dim,
                )
                skipped += 1
                continue

            records.append({
                "chunk_id":        chunk.chunk_id,
                "media_id":        chunk.media_id,
                "start_sec":       float(chunk.start_sec),
                "end_sec":         float(chunk.end_sec),
                "duration_sec":    float(chunk.duration_sec),
                "minio_url":       (chunk.minio_url or "")[:1024],
                "object_key":      (chunk.object_key or "")[:512],
                "transcript":      (chunk.transcript or "")[:2048],
                "language":        (chunk.language or "th")[:8],
                "embedding_model": (chunk.embedding_model or "clap")[:64],
                "created_at":      int(chunk.created_at),
                "embedding":       chunk.embedding.tolist(),
            })

        if not records:
            logger.warning(
                "milvus_service.upsert_nothing",
                total=len(chunks),
                skipped=skipped,
            )
            return 0

        try:
            result = self._client.upsert(
                collection_name=EPISODIC_COLLECTION,
                data=records,
            )
            count = result.get("upsert_count", len(records))
            logger.info(
                "milvus_service.upsert_ok",
                collection=EPISODIC_COLLECTION,
                upserted=count,
                skipped=skipped,
            )
            return count
        except Exception as exc:
            raise StorageError(f"Milvus upsert failed: {exc}") from exc

    def delete_by_media_id(self, media_id: str) -> int:
        """
        Delete all chunks belonging to one media file.

        Returns
        -------
        int
            Number of rows deleted.

        Raises
        ------
        StorageError – Milvus delete failure
        """
        if not media_id:
            raise ValueError("media_id must not be empty")
        try:
            result = self._client.delete(
                collection_name=EPISODIC_COLLECTION,
                filter=f'media_id == "{media_id}"',
            )
            count = result.get("delete_count", 0)
            logger.info(
                "milvus_service.delete_ok",
                media_id=media_id,
                deleted=count,
            )
            return count
        except Exception as exc:
            raise StorageError(
                f"Milvus delete failed for media_id={media_id!r}: {exc}"
            ) from exc

    # ── Search / read operations ──────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        media_id_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Approximate Nearest Neighbour search against the episodic memory.

        Parameters
        ----------
        query_vector:
            Float32 array of dimension `EMBEDDING_DIM`.
        top_k:
            Maximum result count.
        media_id_filter:
            When set, restricts the search to a single media file.
        score_threshold:
            Cosine similarity floor. Hits below this value are dropped.

        Returns
        -------
        list[dict]
            Each dict contains all scalar fields plus `"score"` (cosine similarity).

        Raises
        ------
        ValueError    – wrong vector dimension
        StorageError  – Milvus search failure
        """
        if query_vector.ndim != 1 or query_vector.shape[0] != self._dim:
            raise ValueError(
                f"Expected 1-D vector of dim {self._dim}, "
                f"got shape {query_vector.shape}"
            )

        filter_expr = (
            f'media_id == "{media_id_filter}"' if media_id_filter else None
        )

        try:
            raw_results = self._client.search(
                collection_name=EPISODIC_COLLECTION,
                data=[query_vector.tolist()],
                limit=top_k,
                filter=filter_expr,
                output_fields=_OUTPUT_FIELDS,
                search_params=_SEARCH_PARAMS,
            )
        except Exception as exc:
            raise StorageError(f"Milvus search failed: {exc}") from exc

        hits: list[dict[str, Any]] = []
        for r in raw_results[0]:
            score = float(r.score)
            if score_threshold is not None and score < score_threshold:
                continue
            hit: dict[str, Any] = {
                field: r.entity.get(field) for field in _OUTPUT_FIELDS
            }
            hit["score"] = score
            hits.append(hit)

        logger.debug(
            "milvus_service.search_done",
            returned=len(hits),
            top_k=top_k,
            filter=filter_expr,
        )
        return hits

    def get_by_chunk_id(self, chunk_id: str) -> dict[str, Any] | None:
        """
        Fetch a single record by primary key.

        Returns
        -------
        dict or None
            Scalar fields only (no embedding vector). None if not found.

        Raises
        ------
        StorageError – Milvus read failure
        """
        if not chunk_id:
            raise ValueError("chunk_id must not be empty")
        try:
            results = self._client.get(
                collection_name=EPISODIC_COLLECTION,
                ids=[chunk_id],
                output_fields=_OUTPUT_FIELDS,
            )
            return results[0] if results else None
        except Exception as exc:
            raise StorageError(
                f"Milvus get failed for chunk_id={chunk_id!r}: {exc}"
            ) from exc

    def list_by_media_id(
        self,
        media_id: str,
        limit: int = 1_000,
    ) -> list[dict[str, Any]]:
        """
        Return all chunks for a media file, sorted chronologically.

        Raises
        ------
        StorageError – Milvus query failure
        """
        if not media_id:
            raise ValueError("media_id must not be empty")
        try:
            results = self._client.query(
                collection_name=EPISODIC_COLLECTION,
                filter=f'media_id == "{media_id}"',
                output_fields=_OUTPUT_FIELDS,
                limit=limit,
            )
            return sorted(results, key=lambda r: r.get("start_sec", 0.0))
        except Exception as exc:
            raise StorageError(
                f"Milvus query failed for media_id={media_id!r}: {exc}"
            ) from exc

    def collection_stats(self) -> dict[str, Any]:
        """
        Return basic statistics (row count, index state, etc.).

        Raises
        ------
        StorageError – Milvus stats failure
        """
        try:
            return dict(
                self._client.get_collection_stats(EPISODIC_COLLECTION)
            )
        except Exception as exc:
            raise StorageError(
                f"Failed to retrieve stats for '{EPISODIC_COLLECTION}': {exc}"
            ) from exc
