from typing import Literal

import numpy as np
import structlog

from app.schemas.search import RankedHit
from services.retrieval.reranker import rerank
from services.retrieval.scorer import reciprocal_rank_fusion

logger = structlog.get_logger(__name__)


async def multimodal_search(
    text_query: str,
    audio_url: str | None,
    top_k: int = 10,
    modalities: list[Literal["audio", "visual", "text"]] = ["audio", "visual", "text"],
) -> list[RankedHit]:
    """
    Execute ANN search across requested Milvus collections,
    fuse results with RRF, then rerank with a cross-encoder.
    """
    from storage.milvus.client import get_milvus_client
    from storage.minio.operations import get_presigned_url

    milvus = get_milvus_client()
    candidate_lists: list[list[dict]] = []

    if "text" in modalities:
        from services.embedding.text_encoder import TextEncoder
        vec = TextEncoder().encode(text_query)
        hits = _search_collection(milvus, "text_transcripts", vec, top_k * 2)
        candidate_lists.append(hits)

    if "audio" in modalities:
        from services.embedding.clap_encoder import ClapEncoder
        vecs = ClapEncoder().encode_text([text_query])
        hits = _search_collection(milvus, "audio_segments", vecs[0], top_k * 2)
        candidate_lists.append(hits)

    if "visual" in modalities:
        from services.embedding.visual_encoder import VisualEncoder
        vecs = VisualEncoder().encode_text([text_query])
        hits = _search_collection(milvus, "visual_keyframes", vecs[0], top_k * 2)
        candidate_lists.append(hits)

    if not candidate_lists:
        return []

    merged = reciprocal_rank_fusion(candidate_lists, k=60)[:top_k * 2]
    reranked = rerank(text_query, merged)[:top_k]

    results = []
    for hit in reranked:
        media_url = get_presigned_url(hit.get("object_key", "")) if hit.get("object_key") else None
        results.append(
            RankedHit(
                segment_id=hit["segment_id"],
                media_id=hit["media_id"],
                score=hit["score"],
                start_sec=hit.get("start_sec", 0.0),
                end_sec=hit.get("end_sec", 0.0),
                transcript=hit.get("text"),
                summary=hit.get("summary"),
                media_url=media_url,
            )
        )

    logger.info("searcher.done", text_query=text_query, results=len(results))
    return results


def _search_collection(milvus, collection: str, vec: np.ndarray, limit: int) -> list[dict]:
    results = milvus.search(
        collection_name=collection,
        data=[vec.tolist()],
        limit=limit,
        output_fields=["segment_id", "media_id", "start_sec", "end_sec", "text", "summary", "object_key"],
        search_params={"metric_type": "COSINE", "params": {"ef": 200}},
    )
    hits = []
    for r in results[0]:
        hit = {k: r.entity.get(k) for k in ["segment_id", "media_id", "start_sec", "end_sec", "text", "summary", "object_key"]}
        hit["score"] = float(r.score)
        hits.append(hit)
    return hits
