from __future__ import annotations

import structlog

from app.schemas.search import MomentSearchRequest, MomentSearchResponse
from evaluation.profiles import get_evaluation_profile
from services.retrieval.moments import (
    EvidenceHit,
    evidence_hits_to_video_moments,
    generate_fixed_windows,
)
from storage.milvus.schemas import AUDIO_COLLECTION, TEXT_COLLECTION, VISUAL_COLLECTION

logger = structlog.get_logger(__name__)


class MomentSearchService:
    """
    Canonical long-video search entry point.

    This service owns the benchmark-facing contract:
    Semantic Query + selected media ID -> ranked Video Moments. The current
    implementation is an empty baseline until fixed-window indexing and
    per-modality late fusion are wired in.
    """

    async def run(self, request: MomentSearchRequest) -> MomentSearchResponse:
        profile = get_evaluation_profile(request.profile)
        if request.duration_sec is None:
            logger.info(
                "moment_search.missing_duration",
                media_id=request.media_id,
                query=request.query[:80],
                profile=profile.name,
            )
            return _empty_response(request, profile.name)

        windows = generate_fixed_windows(
            media_id=request.media_id,
            duration_sec=request.duration_sec,
        )
        limit = max(request.top_k * 5, 25)
        visual_hits = _search_weighted_evidence(
            weights=profile.modality_weights,
            source_type="visual",
            search_fn=_search_visual_evidence,
            media_id=request.media_id,
            query=request.query,
            limit=limit,
        )
        asr_hits = _search_weighted_evidence(
            weights=profile.modality_weights,
            source_type="asr",
            search_fn=_search_asr_evidence,
            media_id=request.media_id,
            query=request.query,
            limit=limit,
        )
        audio_hits = _search_weighted_evidence(
            weights=profile.modality_weights,
            source_type="audio",
            search_fn=_search_audio_evidence,
            media_id=request.media_id,
            query=request.query,
            limit=limit,
        )
        moments = evidence_hits_to_video_moments(
            media_id=request.media_id,
            windows=windows,
            hits=[*visual_hits, *asr_hits, *audio_hits],
            top_k=request.top_k,
            source_weights=profile.modality_weights,
        )

        logger.info(
            "moment_search.visual_asr.done",
            media_id=request.media_id,
            query=request.query[:80],
            top_k=request.top_k,
            profile=profile.name,
            visual_hits=len(visual_hits),
            asr_hits=len(asr_hits),
            audio_hits=len(audio_hits),
            moments=len(moments),
        )
        return MomentSearchResponse(
            media_id=request.media_id,
            query=request.query,
            top_k=request.top_k,
            profile=profile.name,
            results=moments,
            total=len(moments),
        )


def _search_weighted_evidence(
    weights: dict,
    source_type: str,
    search_fn,
    media_id: str,
    query: str,
    limit: int,
) -> list[EvidenceHit]:
    if weights.get(source_type, 0.0) <= 0.0:
        return []
    return _search_evidence_safely(
        source_type=source_type,
        search_fn=search_fn,
        media_id=media_id,
        query=query,
        limit=limit,
    )


def _empty_response(request: MomentSearchRequest, profile_name: str) -> MomentSearchResponse:
    return MomentSearchResponse(
        media_id=request.media_id,
        query=request.query,
        top_k=request.top_k,
        profile=profile_name,
        results=[],
        total=0,
    )


def _search_evidence_safely(
    source_type: str,
    search_fn,
    media_id: str,
    query: str,
    limit: int,
) -> list[EvidenceHit]:
    try:
        return search_fn(media_id=media_id, query=query, limit=limit)
    except Exception as exc:
        try:
            from storage.milvus.client import get_milvus_client

            get_milvus_client.cache_clear()
        except Exception:
            pass
        logger.warning(
            "moment_search.evidence_failed",
            source_type=source_type,
            media_id=media_id,
            error=str(exc),
        )
        return []


def _search_visual_evidence(media_id: str, query: str, limit: int) -> list[EvidenceHit]:
    from services.embedding.visual_encoder import VisualEncoder
    from storage.milvus.client import get_milvus_client

    milvus = get_milvus_client()
    query_vector = VisualEncoder().encode_text([query])[0]
    results = milvus.search(
        collection_name=VISUAL_COLLECTION,
        data=[query_vector.tolist()],
        limit=limit,
        filter=f'media_id == "{media_id}"',
        output_fields=["frame_id", "media_id", "timestamp_sec"],
        search_params={"metric_type": "COSINE", "params": {"ef": 200}},
    )

    hits: list[EvidenceHit] = []
    for result in results[0] if results else []:
        entity = result.entity
        result_media_id = entity.get("media_id")
        timestamp_sec = entity.get("timestamp_sec")
        if result_media_id is None or timestamp_sec is None:
            continue
        hits.append(
            EvidenceHit(
                source_type="visual",
                media_id=str(result_media_id),
                score=float(result.score),
                source_id=str(entity.get("frame_id") or ""),
                timestamp_sec=float(timestamp_sec),
            )
        )
    return hits


def _search_asr_evidence(media_id: str, query: str, limit: int) -> list[EvidenceHit]:
    from services.embedding.text_encoder import TextEncoder
    from storage.milvus.client import get_milvus_client

    milvus = get_milvus_client()
    query_vector = TextEncoder().encode(query)
    results = milvus.search(
        collection_name=TEXT_COLLECTION,
        data=[query_vector.tolist()],
        limit=limit,
        filter=f'media_id == "{media_id}"',
        output_fields=["chunk_id", "media_id", "start_sec", "end_sec", "text"],
        search_params={"metric_type": "COSINE", "params": {"ef": 200}},
    )

    hits: list[EvidenceHit] = []
    for result in results[0] if results else []:
        entity = result.entity
        result_media_id = entity.get("media_id")
        start_sec = entity.get("start_sec")
        end_sec = entity.get("end_sec")
        if result_media_id is None or start_sec is None or end_sec is None:
            continue
        hits.append(
            EvidenceHit(
                source_type="asr",
                media_id=str(result_media_id),
                score=float(result.score),
                source_id=str(entity.get("chunk_id") or ""),
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                text=entity.get("text") or None,
            )
        )
    return hits


def _search_audio_evidence(media_id: str, query: str, limit: int) -> list[EvidenceHit]:
    from services.embedding.clap_encoder import ClapEncoder
    from storage.milvus.client import get_milvus_client

    milvus = get_milvus_client()
    query_vector = ClapEncoder().encode_text([query])[0]
    results = milvus.search(
        collection_name=AUDIO_COLLECTION,
        data=[query_vector.tolist()],
        limit=limit,
        filter=f'media_id == "{media_id}"',
        output_fields=["segment_id", "media_id", "start_sec", "end_sec", "summary"],
        search_params={"metric_type": "COSINE", "params": {"ef": 200}},
    )

    hits: list[EvidenceHit] = []
    for result in results[0] if results else []:
        entity = result.entity
        result_media_id = entity.get("media_id")
        start_sec = entity.get("start_sec")
        end_sec = entity.get("end_sec")
        if result_media_id is None or start_sec is None or end_sec is None:
            continue
        hits.append(
            EvidenceHit(
                source_type="audio",
                media_id=str(result_media_id),
                score=float(result.score),
                source_id=str(entity.get("segment_id") or ""),
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                text=entity.get("summary") or None,
            )
        )
    return hits
