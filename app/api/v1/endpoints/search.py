"""
Search endpoints

POST /api/v1/search/episodic  — Primary: text-to-audio semantic search over
                                `csat_episodic_memory` with optional WorldMM
                                LLM reasoning.

POST /api/v1/search/          — Legacy: multimodal ANN search across
                                audio_segments / visual_keyframes / text_transcripts.
"""

from __future__ import annotations

from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException, Query

from app.schemas.search import (
    CollectionMomentSearchRequest,
    EpisodicSearchRequest,
    EpisodicSearchResponse,
    EvaluationQueryListResponse,
    EvaluationQueryOption,
    GroundTruthInterval,
    MomentEvaluationRequest,
    MomentEvaluationResponse,
    MomentEvaluationResult,
    MomentSearchRequest,
    MomentSearchResponse,
    SearchRequest,
    SearchResponse,
)
from evaluation.manifest import EvaluationQuery, EvaluationVideo, iter_evaluation_queries, load_evaluation_manifest
from evaluation.temporal_metrics import RetrievedMoment, temporal_iou
from services.moment_search import CollectionMomentSearchService, MomentSearchService
from services.query_service import QueryService

logger = structlog.get_logger(__name__)
router = APIRouter()


# ── Primary benchmark endpoint: Video Moment search ──────────────────────────

@router.post(
    "/moments",
    response_model=MomentSearchResponse,
    summary="Semantic search over a selected long video",
    description=(
        "Canonical benchmark-facing search contract. Searches within one "
        "selected media item and returns Top-K ranked Video Moments with scores "
        "and source-specific evidence. LLM reasoning is intentionally excluded "
        "from this path."
    ),
)
async def search_moments(request: MomentSearchRequest) -> MomentSearchResponse:
    try:
        response = await MomentSearchService().run(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("search.moments.runtime_error", media_id=request.media_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("search.moments.unhandled", media_id=request.media_id)
        raise HTTPException(status_code=500, detail="Moment search failed unexpectedly.") from exc

    logger.info(
        "search.moments.done",
        media_id=response.media_id,
        query=response.query[:80],
        hits=response.total,
        profile=response.profile,
        top_k=response.top_k,
    )
    return response


@router.post(
    "/moments/collection",
    response_model=MomentSearchResponse,
    summary="Semantic search over an ActivityNet-style video collection",
    description=(
        "Demo-facing collection search. Searches indexed evidence globally, "
        "maps hits back to videos defined by an ActivityNet manifest, and "
        "returns globally ranked timestamped Video Moments."
    ),
)
async def search_moment_collection(
    request: CollectionMomentSearchRequest,
) -> MomentSearchResponse:
    try:
        response = await CollectionMomentSearchService().run(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("search.moments.collection.runtime_error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("search.moments.collection.unhandled")
        raise HTTPException(status_code=500, detail="Collection moment search failed unexpectedly.") from exc

    logger.info(
        "search.moments.collection.done",
        query=response.query[:80],
        hits=response.total,
        profile=response.profile,
        top_k=response.top_k,
    )
    return response


@router.get(
    "/moments/evaluation-queries",
    response_model=EvaluationQueryListResponse,
    summary="List ActivityNet evaluation queries from a manifest",
)
async def list_moment_evaluation_queries(
    manifest_path: str = Query(default="data/manifests/activitynet_dev200_indexed_current.jsonl"),
    limit: int = Query(default=200, ge=1, le=5000),
) -> EvaluationQueryListResponse:
    try:
        videos = load_evaluation_manifest(Path(manifest_path))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    duration_by_media = {video.media_id: video.duration_sec for video in videos}
    queries = [
        EvaluationQueryOption(
            query_id=query.query_id,
            media_id=query.media_id,
            query=query.query,
            duration_sec=duration_by_media.get(query.media_id),
            ground_truth=GroundTruthInterval(
                start_sec=query.ground_truth.start_sec,
                end_sec=query.ground_truth.end_sec,
            ),
        )
        for query in iter_evaluation_queries(videos)[:limit]
    ]
    return EvaluationQueryListResponse(
        manifest_path=manifest_path,
        total=len(queries),
        queries=queries,
    )


@router.post(
    "/moments/evaluate",
    response_model=MomentEvaluationResponse,
    summary="Run moment search for one manifest query and score it against ground truth",
)
async def evaluate_moment_query(request: MomentEvaluationRequest) -> MomentEvaluationResponse:
    try:
        videos = load_evaluation_manifest(Path(request.manifest_path))
        video, query = _find_evaluation_query(videos, request.query_id)
        if video.duration_sec is None:
            raise ValueError(f"media_id={video.media_id} has no duration_sec")
        search_response = await MomentSearchService().run(
            MomentSearchRequest(
                media_id=query.media_id,
                query=query.query,
                top_k=request.top_k,
                duration_sec=video.duration_sec,
                window_sec=request.window_sec,
                stride_sec=request.stride_sec,
                profile=request.profile,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("search.moments.evaluate.runtime_error", query_id=request.query_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("search.moments.evaluate.unhandled", query_id=request.query_id)
        raise HTTPException(status_code=500, detail="Moment evaluation failed unexpectedly.") from exc

    evaluated = []
    best_tiou = 0.0
    hit_rank = None
    for moment in search_response.results:
        score = temporal_iou(
            query.ground_truth,
            RetrievedMoment(
                media_id=moment.media_id,
                start_sec=moment.start_sec,
                end_sec=moment.end_sec,
                score=moment.score,
                moment_id=moment.moment_id,
            ),
        )
        best_tiou = max(best_tiou, score)
        hit = moment.media_id == query.media_id and score >= request.tiou_threshold
        if hit and hit_rank is None:
            hit_rank = moment.rank
        evaluated.append(MomentEvaluationResult(moment=moment, tiou=score, hit=hit))

    return MomentEvaluationResponse(
        query_id=query.query_id,
        media_id=query.media_id,
        query=query.query,
        ground_truth=GroundTruthInterval(
            start_sec=query.ground_truth.start_sec,
            end_sec=query.ground_truth.end_sec,
        ),
        tiou_threshold=request.tiou_threshold,
        hit_rank=hit_rank,
        best_tiou=best_tiou,
        search_response=search_response,
        evaluated_results=evaluated,
    )


def _find_evaluation_query(
    videos: list[EvaluationVideo],
    query_id: str,
) -> tuple[EvaluationVideo, EvaluationQuery]:
    for video in videos:
        for query in video.queries:
            if query.query_id == query_id:
                return video, query
    raise ValueError(f"Unknown query_id: {query_id}")


# ── Primary endpoint: Episodic memory search ──────────────────────────────────

@router.post(
    "/episodic",
    response_model=EpisodicSearchResponse,
    summary="Semantic search over episodic audio memory",
    description=(
        "Encodes the natural-language query into the same vector space used "
        "at ingest time (CLAP or Wav2Vec2), searches the `csat_episodic_memory` "
        "Milvus collection for the Top-K nearest audio chunks, then optionally "
        "runs a WorldMM-style LLM agent to reason over the retrieved context and "
        "pinpoint the exact interaction event."
    ),
)
async def search_episodic(request: EpisodicSearchRequest) -> EpisodicSearchResponse:
    """
    **WorldMM-inspired episodic memory retrieval pipeline:**

    1. Text query → CLAP / sentence-transformer vector  
    2. ANN search in `csat_episodic_memory` (cosine similarity, HNSW)  
    3. Presigned MinIO URLs hydrated on each hit  
    4. (optional) WorldMM prompt → LLM → structured `LLMReasoning` response
    """
    try:
        service = QueryService(embedder_name=request.embedder)
        response = await service.run(
            query=request.query,
            top_k=request.top_k,
            media_id_filter=request.media_id_filter,
            score_threshold=request.score_threshold,
            use_llm=request.use_llm,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("search.episodic.runtime_error", query=request.query)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("search.episodic.unhandled", query=request.query)
        raise HTTPException(status_code=500, detail="Search failed unexpectedly.") from exc

    logger.info(
        "search.episodic.done",
        query=request.query[:80],
        hits=response.total_hits,
        use_llm=request.use_llm,
        embedder=request.embedder,
        top_k=request.top_k,
    )
    return response


# ── Legacy endpoint: Multimodal ANN search ────────────────────────────────────

@router.post(
    "/",
    response_model=SearchResponse,
    summary="Legacy multimodal ANN search",
    description=(
        "Searches across audio_segments, visual_keyframes, and text_transcripts "
        "collections using RRF fusion and cross-encoder reranking. "
        "Prefer `/search/episodic` for CSAT episodic memory retrieval."
    ),
)
async def search_multimodal(request: SearchRequest) -> SearchResponse:
    from services.retrieval.searcher import multimodal_search

    hits = await multimodal_search(
        text_query=request.text_query,
        audio_url=request.audio_url,
        top_k=request.top_k,
        modalities=request.modalities,
    )
    logger.info("search.multimodal.done", query=request.text_query, hits=len(hits))
    return SearchResponse(results=hits, total=len(hits))
