from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import typer

from evaluation.boundary_metrics import evaluate_boundaries
from evaluation.visual_retrieval import (
    recall_at_k,
    rank_visual_candidates,
    temporal_iou,
    temporal_overlap_ratio,
)
from scripts.build_visual_semantic_events import (
    _encode_images,
    _encode_texts,
    _load_model,
    _load_transcript_spans,
    _median_step_ms,
)
from scripts.compare_visual_retrieval import _build_candidates
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
    help="Sweep transcript boundary weights across labeled CASTLE intervals."
)


@app.command()
def main(
    cases_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    transcript_spans: Path = typer.Option(..., exists=True, dir_okay=False),
    output_csv: Path = typer.Option(...),
    output_summary: Path = typer.Option(...),
    weight: list[float] = typer.Option([0.0, 0.1, 0.25, 0.5]),
    model_name: str = typer.Option("ViT-B-32-quickgelu"),
    pretrained: str = typer.Option("openai"),
    batch_size: int = typer.Option(16, min=1),
    context_radius: int = typer.Option(2, min=1),
    boundary_percentile: float = typer.Option(85.0, min=0.0, max=100.0),
    min_boundary_score: float = typer.Option(0.01, min=0.0),
    smoothing_radius: int = typer.Option(1, min=0),
    min_event_sec: float = typer.Option(10.0, min=1.0),
    max_event_sec: float = typer.Option(60.0, min=1.0),
) -> None:
    weights = _validate_weights(weight)
    cases = _load_jsonl(cases_path)
    model, preprocess, tokenizer = _load_model(model_name, pretrained)
    rows: list[dict] = []

    for case in cases:
        frame_dir = Path(case["frame_dir"])
        paths = sorted(frame_dir.glob("*.jpg"))
        if len(paths) < 2:
            raise typer.BadParameter(
                f"case {case['case_id']} requires at least two frames"
            )
        timestamps_ms = [int(path.stem) for path in paths]
        embeddings = _encode_images(model, preprocess, paths, batch_size)
        samples = [
            VisualSample(timestamp_ms=timestamp_ms, embedding=embedding)
            for timestamp_ms, embedding in zip(
                timestamps_ms,
                embeddings,
                strict=True,
            )
        ]
        end_ms = timestamps_ms[-1] + _median_step_ms(timestamps_ms)
        spans = _load_transcript_spans(
            transcript_spans,
            case["video_id"],
        )
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
        references = _load_references(Path(case["reference_path"]))
        queries = _load_jsonl(Path(case["queries_path"]))
        query_embeddings = _encode_texts(
            model,
            tokenizer,
            [query["query"] for query in queries],
        )

        for transcript_weight in weights:
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
            boundary_metrics = evaluate_boundaries(
                predicted_ms=[
                    boundary.timestamp_ms
                    for boundary in segmentation.boundaries
                ],
                reference_ms=references["timestamps_ms"],
                tolerance_ms=references["tolerance_ms"],
            )
            retrieval = _evaluate_retrieval(
                segmentation.intervals,
                timestamps_ms,
                embeddings,
                queries,
                query_embeddings,
            )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "video_id": case["video_id"],
                    "transcript_weight": transcript_weight,
                    **asdict(boundary_metrics),
                    **asdict(segmentation.summary),
                    **retrieval,
                }
            )

    summaries = _summarize(rows, weights)
    _write_csv(output_csv, rows)
    _write_csv(output_summary, summaries)
    typer.echo(
        f"Wrote {len(rows)} case-weight evaluations and "
        f"{len(summaries)} aggregate rows."
    )


def _load_references(path: Path) -> dict:
    rows = _load_jsonl(path)
    tolerances = {int(row["tolerance_ms"]) for row in rows}
    if len(tolerances) != 1:
        raise ValueError(f"{path} must use one boundary tolerance")
    return {
        "timestamps_ms": [
            int(row["timestamp_ms"])
            for row in rows
            if "timestamp_ms" in row
        ],
        "tolerance_ms": tolerances.pop(),
    }


def _evaluate_retrieval(
    intervals,
    timestamps_ms: list[int],
    embeddings: np.ndarray,
    queries: list[dict],
    query_embeddings: np.ndarray,
    *,
    hit_overlap: float = 0.5,
) -> dict:
    candidates = _build_candidates(
        [
            (f"semantic:{index}", interval.start_ms, interval.end_ms)
            for index, interval in enumerate(intervals, start=1)
        ],
        timestamps_ms,
        embeddings,
    )
    hit_ranks: list[int | None] = []
    best_tious: list[float] = []
    top1_tious: list[float] = []
    for query, query_embedding in zip(queries, query_embeddings, strict=True):
        ranked = rank_visual_candidates(query_embedding, candidates)
        hit_rank = None
        best_tiou = 0.0
        top1_tiou = 0.0
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
            best_tiou = max(best_tiou, tiou)
            if rank == 1:
                top1_tiou = tiou
            if hit_rank is None and overlap >= hit_overlap:
                hit_rank = rank
        hit_ranks.append(hit_rank)
        best_tious.append(best_tiou)
        top1_tious.append(top1_tiou)
    query_count = len(queries)
    return {
        "query_count": query_count,
        "retrieval_recall_at_1": recall_at_k(hit_ranks, 1),
        "retrieval_recall_at_3": recall_at_k(hit_ranks, 3),
        "retrieval_recall_at_10": recall_at_k(hit_ranks, 10),
        "retrieval_mrr": sum(
            1 / rank if rank is not None else 0 for rank in hit_ranks
        )
        / query_count,
        "retrieval_mean_best_tiou": float(np.mean(best_tious)),
        "retrieval_mean_top1_tiou": float(np.mean(top1_tious)),
    }


def _summarize(rows: list[dict], weights: list[float]) -> list[dict]:
    summaries = []
    for weight in weights:
        selected = [row for row in rows if row["transcript_weight"] == weight]
        predicted = sum(row["predicted_count"] for row in selected)
        reference = sum(row["reference_count"] for row in selected)
        matched = sum(row["matched_count"] for row in selected)
        precision = matched / predicted if predicted else 0.0
        recall = matched / reference if reference else 0.0
        summaries.append(
            {
                "transcript_weight": weight,
                "case_count": len(selected),
                "boundary_precision_micro": precision,
                "boundary_recall_micro": recall,
                "boundary_f1_micro": (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                ),
                "boundary_f1_macro": float(
                    np.mean([row["f1"] for row in selected])
                ),
                "mean_event_count": float(
                    np.mean([row["event_count"] for row in selected])
                ),
                "mean_event_duration_ms": float(
                    np.mean([row["mean_duration_ms"] for row in selected])
                ),
                "retrieval_recall_at_1": float(
                    np.mean(
                        [row["retrieval_recall_at_1"] for row in selected]
                    )
                ),
                "retrieval_recall_at_3": float(
                    np.mean(
                        [row["retrieval_recall_at_3"] for row in selected]
                    )
                ),
                "retrieval_recall_at_10": float(
                    np.mean(
                        [row["retrieval_recall_at_10"] for row in selected]
                    )
                ),
                "retrieval_mrr": float(
                    np.mean([row["retrieval_mrr"] for row in selected])
                ),
                "retrieval_mean_best_tiou": float(
                    np.mean(
                        [row["retrieval_mean_best_tiou"] for row in selected]
                    )
                ),
                "retrieval_mean_top1_tiou": float(
                    np.mean(
                        [row["retrieval_mean_top1_tiou"] for row in selected]
                    )
                ),
            }
        )
    return summaries


def _validate_weights(weights: list[float]) -> list[float]:
    if not weights:
        raise ValueError("at least one transcript weight is required")
    if any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("transcript weights must be between 0 and 1")
    return list(dict.fromkeys(weights))


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
