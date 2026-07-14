from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import typer

from evaluation.visual_retrieval import (
    VisualCandidate,
    rank_visual_candidates,
    recall_at_k,
    reciprocal_rank_fuse_visual_candidates,
    temporal_iou,
    temporal_overlap_ratio,
    temporal_precision,
)
from scripts.build_visual_semantic_events import (
    _encode_images,
    _encode_texts,
    _load_model,
    _load_transcript_spans,
    _median_step_ms,
)
from scripts.compare_visual_retrieval import _build_candidates, _fixed_intervals
from services.events.transcript_boundaries import (
    contextual_transcript_distances,
    transcript_text_bins,
)
from services.events.visual_boundaries import VisualSample
from services.events.visual_segmentation import (
    VisualSegmentationConfig,
    run_visual_segmentation,
)

app = typer.Typer(
    help=(
        "Compare event-first direct-visual retrieval against fixed CASTLE "
        "windows across labeled development cases."
    )
)


@app.command()
def main(
    cases_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    transcript_spans: Path = typer.Option(..., exists=True, dir_okay=False),
    output_cases: Path = typer.Option(...),
    output_summary: Path = typer.Option(...),
    transcript_weight: float = typer.Option(0.25, min=0.0, max=1.0),
    model_name: str = typer.Option("ViT-B-32-quickgelu"),
    pretrained: str = typer.Option("openai"),
    batch_size: int = typer.Option(16, min=1),
    context_radius: int = typer.Option(2, min=1),
    boundary_percentile: float = typer.Option(85.0, min=0.0, max=100.0),
    min_boundary_score: float = typer.Option(0.01, min=0.0),
    smoothing_radius: int = typer.Option(1, min=0),
    min_event_sec: float = typer.Option(10.0, min=1.0),
    max_event_sec: float = typer.Option(60.0, min=1.0),
    hit_overlap: float = typer.Option(0.5, min=0.0, max=1.0),
    refinement_overlap: float = typer.Option(0.8, min=0.0, max=1.0),
    rrf_k: int = typer.Option(60, min=1),
    semantic_weight: float = typer.Option(1.0, min=0.0),
    fixed_120s_weight: float = typer.Option(0.85, min=0.0),
    transcript_rerank_weight: float = typer.Option(0.05, min=0.0),
    transcript_semantic_weight: float = typer.Option(0.025, min=0.0),
    heart_rate_rerank_weight: float = typer.Option(0.1, min=0.0),
) -> None:
    cases = _load_jsonl(cases_path)
    model, preprocess, tokenizer = _load_model(model_name, pretrained)
    case_rows: list[dict] = []
    aggregate_inputs: dict[str, list[dict]] = {}

    for case in cases:
        frame_dir = Path(case["frame_dir"])
        paths = sorted(frame_dir.glob("*.jpg"))
        if len(paths) < 2:
            raise typer.BadParameter(
                f"case {case['case_id']} requires at least two frames"
            )
        timestamps_ms = [int(path.stem) for path in paths]
        embeddings = _encode_images(model, preprocess, paths, batch_size)
        end_ms = timestamps_ms[-1] + _median_step_ms(timestamps_ms)
        samples = [
            VisualSample(timestamp_ms=timestamp_ms, embedding=embedding)
            for timestamp_ms, embedding in zip(
                timestamps_ms,
                embeddings,
                strict=True,
            )
        ]
        semantic_intervals = _semantic_intervals(
            case=case,
            samples=samples,
            timestamps_ms=timestamps_ms,
            end_ms=end_ms,
            embeddings=embeddings,
            tokenizer=tokenizer,
            model=model,
            transcript_spans=transcript_spans,
            transcript_weight=transcript_weight,
            context_radius=context_radius,
            boundary_percentile=boundary_percentile,
            min_boundary_score=min_boundary_score,
            smoothing_radius=smoothing_radius,
            min_event_sec=min_event_sec,
            max_event_sec=max_event_sec,
        )
        interval_sets = {
            f"semantic_visual_text_w{transcript_weight:g}": semantic_intervals,
            "fixed_30s": _fixed_intervals(
                timestamps_ms[0],
                end_ms,
                window_ms=30_000,
                stride_ms=20_000,
            ),
            "fixed_120s": _fixed_intervals(
                timestamps_ms[0],
                end_ms,
                window_ms=120_000,
                stride_ms=120_000,
            ),
        }
        semantic_label = f"semantic_visual_text_w{transcript_weight:g}"
        spans = _load_transcript_spans(transcript_spans, case["video_id"])
        queries = _load_jsonl(Path(case["queries_path"]))
        query_embeddings = _encode_texts(
            model,
            tokenizer,
            [query["query"] for query in queries],
        )
        candidates_by_type = {
            candidate_type: _build_candidates(
                intervals,
                timestamps_ms,
                embeddings,
            )
            for candidate_type, intervals in interval_sets.items()
        }
        transcript_by_candidate = {
            candidate_type: _candidate_transcripts(intervals, spans)
            for candidate_type, intervals in interval_sets.items()
        }
        transcript_embedding_by_candidate = {
            candidate_type: _encode_texts(
                model,
                tokenizer,
                [
                    transcript_by_candidate[candidate_type].get(
                        candidate.candidate_id,
                        "",
                    )
                    for candidate in candidates
                ],
            )
            for candidate_type, candidates in candidates_by_type.items()
        }

        for candidate_type, candidates in candidates_by_type.items():
            per_query = _evaluate_queries(
                candidates,
                queries,
                query_embeddings,
                hit_overlap=hit_overlap,
            )
            aggregate_inputs.setdefault(candidate_type, []).extend(per_query)
            case_rows.append(
                {
                    "case_id": case["case_id"],
                    "video_id": case["video_id"],
                    "candidate_type": candidate_type,
                    "query_count": len(queries),
                    "candidate_count": len(candidates),
                    "mean_candidate_duration_ms": _mean_duration(
                        interval_sets[candidate_type]
                    ),
                    **_summarize_query_metrics(per_query),
                }
            )
            transcript_label = f"{candidate_type}_transcript_rerank"
            transcript_per_query = _evaluate_transcript_reranked_queries(
                candidates,
                transcript_by_candidate[candidate_type],
                transcript_embedding_by_candidate[candidate_type],
                queries,
                query_embeddings,
                hit_overlap=hit_overlap,
                transcript_rerank_weight=transcript_rerank_weight,
                transcript_semantic_weight=transcript_semantic_weight,
                heart_rate_rerank_weight=heart_rate_rerank_weight,
            )
            aggregate_inputs.setdefault(transcript_label, []).extend(
                transcript_per_query
            )
            case_rows.append(
                {
                    "case_id": case["case_id"],
                    "video_id": case["video_id"],
                    "candidate_type": transcript_label,
                    "query_count": len(queries),
                    "candidate_count": len(candidates),
                    "mean_candidate_duration_ms": _mean_duration(
                        interval_sets[candidate_type]
                    ),
                    **_summarize_query_metrics(transcript_per_query),
                }
            )
        fused_label = "fused_semantic_fixed120_rrf"
        fused_per_query = _evaluate_fused_queries(
            candidates_by_type[semantic_label],
            candidates_by_type["fixed_120s"],
            queries,
            query_embeddings,
            hit_overlap=hit_overlap,
            refinement_overlap=None,
            rrf_k=rrf_k,
            semantic_weight=semantic_weight,
            fixed_120s_weight=fixed_120s_weight,
        )
        aggregate_inputs.setdefault(fused_label, []).extend(fused_per_query)
        case_rows.append(
            {
                "case_id": case["case_id"],
                "video_id": case["video_id"],
                "candidate_type": fused_label,
                "query_count": len(queries),
                "candidate_count": (
                    len(candidates_by_type[semantic_label])
                    + len(candidates_by_type["fixed_120s"])
                ),
                "mean_candidate_duration_ms": _mean_duration(
                    [
                        *interval_sets[semantic_label],
                        *interval_sets["fixed_120s"],
                    ]
                ),
                **_summarize_query_metrics(fused_per_query),
            }
        )
        refined_fused_label = "fused_semantic_fixed120_rrf_semantic_refined"
        refined_fused_per_query = _evaluate_fused_queries(
            candidates_by_type[semantic_label],
            candidates_by_type["fixed_120s"],
            queries,
            query_embeddings,
            hit_overlap=hit_overlap,
            refinement_overlap=refinement_overlap,
            rrf_k=rrf_k,
            semantic_weight=semantic_weight,
            fixed_120s_weight=fixed_120s_weight,
        )
        aggregate_inputs.setdefault(refined_fused_label, []).extend(
            refined_fused_per_query
        )
        case_rows.append(
            {
                "case_id": case["case_id"],
                "video_id": case["video_id"],
                "candidate_type": refined_fused_label,
                "query_count": len(queries),
                "candidate_count": (
                    len(candidates_by_type[semantic_label])
                    + len(candidates_by_type["fixed_120s"])
                ),
                "mean_candidate_duration_ms": _mean_duration(
                    [
                        *interval_sets[semantic_label],
                        *interval_sets["fixed_120s"],
                    ]
                ),
                **_summarize_query_metrics(refined_fused_per_query),
            }
        )
        fused_transcript_label = "fused_semantic_fixed120_rrf_transcript_hr_gated"
        fused_transcript_per_query = _evaluate_fused_transcript_queries(
            candidates_by_type[semantic_label],
            candidates_by_type["fixed_120s"],
            {
                **transcript_by_candidate[semantic_label],
                **transcript_by_candidate["fixed_120s"],
            },
            _encode_texts(
                model,
                tokenizer,
                [
                    {
                        **transcript_by_candidate[semantic_label],
                        **transcript_by_candidate["fixed_120s"],
                    }.get(candidate.candidate_id, "")
                    for candidate in [
                        *candidates_by_type[semantic_label],
                        *candidates_by_type["fixed_120s"],
                    ]
                ],
            ),
            queries,
            query_embeddings,
            hit_overlap=hit_overlap,
            rrf_k=rrf_k,
            semantic_weight=semantic_weight,
            fixed_120s_weight=fixed_120s_weight,
            transcript_rerank_weight=transcript_rerank_weight,
            transcript_semantic_weight=transcript_semantic_weight,
            heart_rate_rerank_weight=heart_rate_rerank_weight,
        )
        aggregate_inputs.setdefault(fused_transcript_label, []).extend(
            fused_transcript_per_query
        )
        case_rows.append(
            {
                "case_id": case["case_id"],
                "video_id": case["video_id"],
                "candidate_type": fused_transcript_label,
                "query_count": len(queries),
                "candidate_count": (
                    len(candidates_by_type[semantic_label])
                    + len(candidates_by_type["fixed_120s"])
                ),
                "mean_candidate_duration_ms": _mean_duration(
                    [
                        *interval_sets[semantic_label],
                        *interval_sets["fixed_120s"],
                    ]
                ),
                **_summarize_query_metrics(fused_transcript_per_query),
            }
        )

    summary_rows = [
        {
            "candidate_type": candidate_type,
            "query_count": len(rows),
            **_summarize_query_metrics(rows),
        }
        for candidate_type, rows in aggregate_inputs.items()
    ]
    _write_csv(output_cases, case_rows)
    _write_csv(output_summary, summary_rows)
    typer.echo(
        f"Wrote {len(case_rows)} case rows and {len(summary_rows)} summary rows."
    )


def _semantic_intervals(
    *,
    case: dict,
    samples: list[VisualSample],
    timestamps_ms: list[int],
    end_ms: int,
    embeddings: np.ndarray,
    tokenizer,
    model,
    transcript_spans: Path,
    transcript_weight: float,
    context_radius: int,
    boundary_percentile: float,
    min_boundary_score: float,
    smoothing_radius: int,
    min_event_sec: float,
    max_event_sec: float,
) -> list[tuple[str, int, int]]:
    spans = _load_transcript_spans(transcript_spans, case["video_id"])
    texts = transcript_text_bins(
        timestamps_ms,
        end_ms=end_ms,
        spans=spans,
    )
    text_embeddings = _encode_texts(model, tokenizer, texts)
    transcript = contextual_transcript_distances(
        text_embeddings,
        np.asarray([bool(text) for text in texts]),
        context_radius=context_radius,
    )
    segmentation = run_visual_segmentation(
        samples,
        VisualSegmentationConfig(
            name=f"v2-text-w{transcript_weight:g}",
            detector="v2",
            context_radius=context_radius,
            smoothing_radius=smoothing_radius,
            boundary_percentile=boundary_percentile,
            min_boundary_score=min_boundary_score,
            min_event_ms=round(min_event_sec * 1000),
            max_event_ms=round(max_event_sec * 1000),
        ),
        start_ms=timestamps_ms[0],
        end_ms=end_ms,
        transcript_scores=transcript.scores,
        transcript_available=transcript.available,
        transcript_weight=transcript_weight,
    )
    return [
        (f"semantic:{index}", interval.start_ms, interval.end_ms)
        for index, interval in enumerate(segmentation.intervals, start=1)
    ]


def _evaluate_queries(
    candidates: list[VisualCandidate],
    queries: list[dict],
    query_embeddings: np.ndarray,
    *,
    hit_overlap: float,
) -> list[dict]:
    rows = []
    for query, query_embedding in zip(queries, query_embeddings, strict=True):
        ranked = rank_visual_candidates(query_embedding, candidates)
        hit_rank = None
        best_overlap = 0.0
        best_tiou = 0.0
        best_temporal_precision = 0.0
        best_tiou_duration_ms = 0
        top1_tiou = 0.0
        top1_temporal_precision = 0.0
        top1_duration_ms = 0
        for rank, (candidate, _score) in enumerate(ranked, start=1):
            overlap = temporal_overlap_ratio(
                candidate.start_ms,
                candidate.end_ms,
                query["expected_start_ms"],
                query["expected_end_ms"],
            )
            tiou = temporal_iou(
                candidate.start_ms,
                candidate.end_ms,
                query["expected_start_ms"],
                query["expected_end_ms"],
            )
            precision = temporal_precision(
                candidate.start_ms,
                candidate.end_ms,
                query["expected_start_ms"],
                query["expected_end_ms"],
            )
            best_overlap = max(best_overlap, overlap)
            if tiou > best_tiou:
                best_tiou = tiou
                best_temporal_precision = precision
                best_tiou_duration_ms = candidate.end_ms - candidate.start_ms
            if rank == 1:
                top1_tiou = tiou
                top1_temporal_precision = precision
                top1_duration_ms = candidate.end_ms - candidate.start_ms
            if hit_rank is None and overlap >= hit_overlap:
                hit_rank = rank
        rows.append(
            {
                "hit_rank": hit_rank,
                "best_overlap": best_overlap,
                "best_tiou": best_tiou,
                "best_temporal_precision": best_temporal_precision,
                "best_tiou_duration_ms": best_tiou_duration_ms,
                "top1_tiou": top1_tiou,
                "top1_temporal_precision": top1_temporal_precision,
                "top1_duration_ms": top1_duration_ms,
            }
        )
    return rows


def _evaluate_fused_queries(
    semantic_candidates: list[VisualCandidate],
    fixed_candidates: list[VisualCandidate],
    queries: list[dict],
    query_embeddings: np.ndarray,
    *,
    hit_overlap: float,
    refinement_overlap: float | None,
    rrf_k: int,
    semantic_weight: float,
    fixed_120s_weight: float,
) -> list[dict]:
    rows = []
    for query, query_embedding in zip(queries, query_embeddings, strict=True):
        semantic_ranked = rank_visual_candidates(
            query_embedding,
            semantic_candidates,
        )
        fixed_ranked = rank_visual_candidates(query_embedding, fixed_candidates)
        fused = reciprocal_rank_fuse_visual_candidates(
            [semantic_ranked, fixed_ranked],
            k=rrf_k,
            weights=[semantic_weight, fixed_120s_weight],
        )
        if refinement_overlap is not None:
            fused = _prefer_overlapping_semantic_candidates(
                fused,
                semantic_ranked,
                min_semantic_coverage=refinement_overlap,
            )
        rows.append(
            _evaluate_ranked_query(
                fused,
                query,
                hit_overlap=hit_overlap,
            )
        )
    return rows


def _prefer_overlapping_semantic_candidates(
    ranked: list[tuple[VisualCandidate, float]],
    semantic_ranked: list[tuple[VisualCandidate, float]],
    *,
    min_semantic_coverage: float,
) -> list[tuple[VisualCandidate, float]]:
    refined: list[tuple[VisualCandidate, float]] = []
    used: set[str] = set()
    semantic_by_id = {
        candidate.candidate_id: (candidate, score)
        for candidate, score in semantic_ranked
    }

    for candidate, score in ranked:
        replacement = None
        if candidate.candidate_id.startswith("fixed:"):
            replacement = _best_semantic_inside_candidate(
                candidate,
                semantic_ranked,
                min_semantic_coverage=min_semantic_coverage,
            )
        if replacement is not None:
            selected_candidate, selected_score = replacement
            if selected_candidate.candidate_id not in used:
                refined.append((selected_candidate, selected_score))
                used.add(selected_candidate.candidate_id)
        if candidate.candidate_id in used:
            continue
        refined.append((candidate, score))
        used.add(candidate.candidate_id)

    for candidate, score in semantic_by_id.values():
        if candidate.candidate_id not in used:
            refined.append((candidate, score))
            used.add(candidate.candidate_id)

    return refined


def _best_semantic_inside_candidate(
    fixed_candidate: VisualCandidate,
    semantic_ranked: list[tuple[VisualCandidate, float]],
    *,
    min_semantic_coverage: float,
) -> tuple[VisualCandidate, float] | None:
    for semantic_candidate, semantic_score in semantic_ranked:
        semantic_coverage = temporal_overlap_ratio(
            fixed_candidate.start_ms,
            fixed_candidate.end_ms,
            semantic_candidate.start_ms,
            semantic_candidate.end_ms,
        )
        if semantic_coverage >= min_semantic_coverage:
            return semantic_candidate, semantic_score
    return None


def _evaluate_transcript_reranked_queries(
    candidates: list[VisualCandidate],
    transcript_by_candidate: dict[str, str],
    transcript_embeddings: np.ndarray,
    queries: list[dict],
    query_embeddings: np.ndarray,
    *,
    hit_overlap: float,
    transcript_rerank_weight: float,
    transcript_semantic_weight: float,
    heart_rate_rerank_weight: float,
) -> list[dict]:
    embedding_by_candidate = {
        candidate.candidate_id: embedding
        for candidate, embedding in zip(
            candidates,
            transcript_embeddings,
            strict=True,
        )
    }
    rows = []
    for query, query_embedding in zip(queries, query_embeddings, strict=True):
        ranked = rank_visual_candidates(query_embedding, candidates)
        reranked = _rerank_with_auxiliary_evidence(
            ranked,
            query=query["query"],
            query_embedding=query_embedding,
            transcript_by_candidate=transcript_by_candidate,
            transcript_embedding_by_candidate=embedding_by_candidate,
            transcript_rerank_weight=transcript_rerank_weight,
            transcript_semantic_weight=transcript_semantic_weight,
            heart_rate_rerank_weight=heart_rate_rerank_weight,
        )
        rows.append(
            _evaluate_ranked_query(
                reranked,
                query,
                hit_overlap=hit_overlap,
            )
        )
    return rows


def _evaluate_fused_transcript_queries(
    semantic_candidates: list[VisualCandidate],
    fixed_candidates: list[VisualCandidate],
    transcript_by_candidate: dict[str, str],
    transcript_embeddings: np.ndarray,
    queries: list[dict],
    query_embeddings: np.ndarray,
    *,
    hit_overlap: float,
    rrf_k: int,
    semantic_weight: float,
    fixed_120s_weight: float,
    transcript_rerank_weight: float,
    transcript_semantic_weight: float,
    heart_rate_rerank_weight: float,
) -> list[dict]:
    all_candidates = [*semantic_candidates, *fixed_candidates]
    embedding_by_candidate = {
        candidate.candidate_id: embedding
        for candidate, embedding in zip(
            all_candidates,
            transcript_embeddings,
            strict=True,
        )
    }
    rows = []
    for query, query_embedding in zip(queries, query_embeddings, strict=True):
        semantic_ranked = rank_visual_candidates(
            query_embedding,
            semantic_candidates,
        )
        fixed_ranked = rank_visual_candidates(query_embedding, fixed_candidates)
        fused = reciprocal_rank_fuse_visual_candidates(
            [semantic_ranked, fixed_ranked],
            k=rrf_k,
            weights=[semantic_weight, fixed_120s_weight],
        )
        reranked = _rerank_with_auxiliary_evidence(
            fused,
            query=query["query"],
            query_embedding=query_embedding,
            transcript_by_candidate=transcript_by_candidate,
            transcript_embedding_by_candidate=embedding_by_candidate,
            transcript_rerank_weight=transcript_rerank_weight,
            transcript_semantic_weight=transcript_semantic_weight,
            heart_rate_rerank_weight=heart_rate_rerank_weight,
        )
        rows.append(
            _evaluate_ranked_query(
                reranked,
                query,
                hit_overlap=hit_overlap,
            )
        )
    return rows


def _rerank_with_auxiliary_evidence(
    ranked: list[tuple[VisualCandidate, float]],
    *,
    query: str,
    query_embedding: np.ndarray,
    transcript_by_candidate: dict[str, str],
    transcript_embedding_by_candidate: dict[str, np.ndarray],
    transcript_rerank_weight: float,
    transcript_semantic_weight: float,
    heart_rate_rerank_weight: float,
) -> list[tuple[VisualCandidate, float]]:
    scored = []
    for rank, (candidate, score) in enumerate(ranked, start=1):
        transcript = transcript_by_candidate.get(candidate.candidate_id, "")
        transcript_embedding = transcript_embedding_by_candidate.get(
            candidate.candidate_id
        )
        auxiliary_score = (
            transcript_rerank_weight * _lexical_overlap(query, transcript)
        )
        if transcript_embedding is not None:
            auxiliary_score += transcript_semantic_weight * float(
                np.dot(query_embedding, transcript_embedding)
            )
        if _is_heart_rate_query(query):
            auxiliary_score += heart_rate_rerank_weight * _duration_activity_prior(
                candidate
            )
        scored.append(
            (
                candidate,
                _rank_base_score(score, rank) + auxiliary_score,
            )
        )
    return sorted(scored, key=lambda item: item[1], reverse=True)


def _evaluate_ranked_query(
    ranked: list[tuple[VisualCandidate, float]],
    query: dict,
    *,
    hit_overlap: float,
) -> dict:
    hit_rank = None
    best_overlap = 0.0
    best_tiou = 0.0
    best_temporal_precision = 0.0
    best_tiou_duration_ms = 0
    top1_tiou = 0.0
    top1_temporal_precision = 0.0
    top1_duration_ms = 0
    for rank, (candidate, _score) in enumerate(ranked, start=1):
        overlap = temporal_overlap_ratio(
            candidate.start_ms,
            candidate.end_ms,
            query["expected_start_ms"],
            query["expected_end_ms"],
        )
        tiou = temporal_iou(
            candidate.start_ms,
            candidate.end_ms,
            query["expected_start_ms"],
            query["expected_end_ms"],
        )
        precision = temporal_precision(
            candidate.start_ms,
            candidate.end_ms,
            query["expected_start_ms"],
            query["expected_end_ms"],
        )
        best_overlap = max(best_overlap, overlap)
        if tiou > best_tiou:
            best_tiou = tiou
            best_temporal_precision = precision
            best_tiou_duration_ms = candidate.end_ms - candidate.start_ms
        if rank == 1:
            top1_tiou = tiou
            top1_temporal_precision = precision
            top1_duration_ms = candidate.end_ms - candidate.start_ms
        if hit_rank is None and overlap >= hit_overlap:
            hit_rank = rank
    return {
        "hit_rank": hit_rank,
        "best_overlap": best_overlap,
        "best_tiou": best_tiou,
        "best_temporal_precision": best_temporal_precision,
        "best_tiou_duration_ms": best_tiou_duration_ms,
        "top1_tiou": top1_tiou,
        "top1_temporal_precision": top1_temporal_precision,
        "top1_duration_ms": top1_duration_ms,
    }


def _summarize_query_metrics(rows: list[dict]) -> dict:
    hit_ranks = [row["hit_rank"] for row in rows]
    return {
        "Recall@1": recall_at_k(hit_ranks, 1),
        "Recall@3": recall_at_k(hit_ranks, 3),
        "Recall@10": recall_at_k(hit_ranks, 10),
        "MRR": sum(1 / rank if rank is not None else 0 for rank in hit_ranks)
        / len(rows),
        "mean_best_overlap": float(np.mean([row["best_overlap"] for row in rows])),
        "mean_best_tIoU": float(np.mean([row["best_tiou"] for row in rows])),
        "mean_best_temporal_precision": float(
            np.mean([row["best_temporal_precision"] for row in rows])
        ),
        "mean_best_tIoU_duration_ms": float(
            np.mean([row["best_tiou_duration_ms"] for row in rows])
        ),
        "mean_top1_tIoU": float(np.mean([row["top1_tiou"] for row in rows])),
        "mean_top1_temporal_precision": float(
            np.mean([row["top1_temporal_precision"] for row in rows])
        ),
        "mean_top1_duration_ms": float(
            np.mean([row["top1_duration_ms"] for row in rows])
        ),
    }


def _mean_duration(intervals: list[tuple[str, int, int]]) -> float:
    if not intervals:
        return 0.0
    return float(np.mean([end_ms - start_ms for _id, start_ms, end_ms in intervals]))


def _candidate_transcripts(
    intervals: list[tuple[str, int, int]],
    spans,
) -> dict[str, str]:
    transcripts = {}
    for candidate_id, start_ms, end_ms in intervals:
        overlapping = [
            span.text
            for span in spans
            if span.start_ms < end_ms and span.end_ms > start_ms
        ]
        transcripts[candidate_id] = " ".join(overlapping)
    return transcripts


def _rank_base_score(score: float, rank: int) -> float:
    if score < 0.05:
        return 1.0 / (rank + 1)
    return score


def _lexical_overlap(query: str, text: str) -> float:
    query_terms = _content_terms(query)
    if not query_terms:
        return 0.0
    text_terms = set(_content_terms(text))
    if not text_terms:
        return 0.0
    return len(set(query_terms) & text_terms) / len(set(query_terms))


def _content_terms(text: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "at",
        "beside",
        "in",
        "inside",
        "of",
        "on",
        "one",
        "the",
        "to",
        "with",
    }
    return [
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if len(term) > 2 and term not in stopwords
    ]


def _is_heart_rate_query(query: str) -> bool:
    terms = set(_content_terms(query))
    heart_rate_terms = {
        "active",
        "activity",
        "arousal",
        "exertion",
        "heart",
        "intense",
        "pulse",
        "resting",
        "running",
        "stress",
        "walking",
    }
    return bool(terms & heart_rate_terms)


def _duration_activity_prior(candidate: VisualCandidate) -> float:
    duration_ms = candidate.end_ms - candidate.start_ms
    if duration_ms <= 0:
        return 0.0
    return min(duration_ms / 120_000, 1.0)


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    app()
