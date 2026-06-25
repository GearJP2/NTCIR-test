from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import typer
from PIL import Image

from evaluation.visual_retrieval import (
    VisualCandidate,
    rank_visual_candidates,
    temporal_iou,
    temporal_overlap_ratio,
)
from services.events.manifest import load_event_manifest

app = typer.Typer(
    help="Compare direct-visual retrieval over semantic and fixed CASTLE events."
)


@app.command()
def main(
    frame_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    semantic_manifest: Path = typer.Argument(..., exists=True, dir_okay=False),
    queries_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_results: Path = typer.Option(...),
    output_summary: Path = typer.Option(...),
    model_name: str = typer.Option("ViT-B-32-quickgelu"),
    pretrained: str = typer.Option("openai"),
    batch_size: int = typer.Option(8, min=1),
    hit_overlap: float = typer.Option(0.5, min=0.0, max=1.0),
) -> None:
    paths = sorted(frame_dir.glob("*.jpg"))
    timestamps_ms = [int(path.stem) for path in paths]
    model, preprocess, tokenizer = _load_model(model_name, pretrained)
    frame_embeddings = _encode_images(model, preprocess, paths, batch_size)
    sample_end_ms = timestamps_ms[-1] + _median_step(timestamps_ms)

    semantic_records = [
        record
        for record in load_event_manifest(semantic_manifest)
        if record.event_kind.value == "semantic_micro"
    ]
    semantic_label = (
        "semantic_visual_text"
        if any(record.coverage.transcript for record in semantic_records)
        else "semantic_v2"
    )
    interval_sets = {
        semantic_label: [
            (record.event_id, record.start_ms, record.end_ms)
            for record in semantic_records
        ],
        "fixed_30s": _fixed_intervals(
            timestamps_ms[0],
            sample_end_ms,
            window_ms=30_000,
            stride_ms=20_000,
        ),
        "fixed_120s": _fixed_intervals(
            timestamps_ms[0],
            sample_end_ms,
            window_ms=120_000,
            stride_ms=120_000,
        ),
    }
    candidates_by_type = {
        candidate_type: _build_candidates(
            intervals,
            timestamps_ms,
            frame_embeddings,
        )
        for candidate_type, intervals in interval_sets.items()
    }
    queries = [
        json.loads(line)
        for line in queries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    query_embeddings = _encode_texts(
        model,
        tokenizer,
        [query["query"] for query in queries],
    )

    result_rows: list[dict] = []
    summary_accumulator: defaultdict[str, list[dict]] = defaultdict(list)
    for query, query_embedding in zip(queries, query_embeddings, strict=True):
        for candidate_type, candidates in candidates_by_type.items():
            ranked = rank_visual_candidates(query_embedding, candidates)
            hit_rank = None
            best_overlap = 0.0
            best_tiou = 0.0
            top1_tiou = 0.0
            for rank, (candidate, score) in enumerate(ranked, start=1):
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
                best_overlap = max(best_overlap, overlap)
                best_tiou = max(best_tiou, tiou)
                if rank == 1:
                    top1_tiou = tiou
                if hit_rank is None and overlap >= hit_overlap:
                    hit_rank = rank
                result_rows.append(
                    {
                        "query_id": query["query_id"],
                        "query": query["query"],
                        "candidate_type": candidate_type,
                        "rank": rank,
                        "candidate_id": candidate.candidate_id,
                        "start_ms": candidate.start_ms,
                        "end_ms": candidate.end_ms,
                        "score": score,
                        "expected_overlap": overlap,
                        "temporal_iou": tiou,
                        "hit": overlap >= hit_overlap,
                    }
                )
            summary_accumulator[candidate_type].append(
                {
                    "hit_rank": hit_rank,
                    "best_overlap": best_overlap,
                    "best_tiou": best_tiou,
                    "top1_tiou": top1_tiou,
                }
            )

    summary_rows = []
    for candidate_type, rows in summary_accumulator.items():
        summary_rows.append(
            {
                "candidate_type": candidate_type,
                "query_count": len(rows),
                "Recall@1": sum(row["hit_rank"] == 1 for row in rows) / len(rows),
                "Recall@3": sum(
                    row["hit_rank"] is not None and row["hit_rank"] <= 3
                    for row in rows
                )
                / len(rows),
                "MRR": sum(
                    1 / row["hit_rank"] if row["hit_rank"] is not None else 0
                    for row in rows
                )
                / len(rows),
                "mean_best_overlap": sum(row["best_overlap"] for row in rows)
                / len(rows),
                "mean_best_tIoU": sum(row["best_tiou"] for row in rows)
                / len(rows),
                "mean_top1_tIoU": sum(row["top1_tiou"] for row in rows)
                / len(rows),
            }
        )

    _write_csv(output_results, result_rows)
    _write_csv(output_summary, summary_rows)
    typer.echo(f"Wrote retrieval comparison for {len(queries)} visual queries.")


def _load_model(model_name: str, pretrained: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        cache_dir="model_cache",
    )
    return model.to("cpu").eval(), preprocess, open_clip.get_tokenizer(model_name)


def _encode_images(model, preprocess, paths: list[Path], batch_size: int) -> np.ndarray:
    import torch

    batches = []
    for offset in range(0, len(paths), batch_size):
        batch = torch.stack(
            [
                preprocess(Image.open(path).convert("RGB"))
                for path in paths[offset : offset + batch_size]
            ]
        )
        with torch.no_grad():
            embeddings = model.encode_image(batch)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
        batches.append(embeddings.cpu().numpy().astype(np.float32))
    return np.concatenate(batches)


def _encode_texts(model, tokenizer, queries: list[str]) -> np.ndarray:
    import torch

    tokens = tokenizer(queries)
    with torch.no_grad():
        embeddings = model.encode_text(tokens)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
    return embeddings.cpu().numpy().astype(np.float32)


def _fixed_intervals(
    start_ms: int,
    end_ms: int,
    *,
    window_ms: int,
    stride_ms: int,
) -> list[tuple[str, int, int]]:
    intervals = []
    cursor = start_ms
    index = 1
    while cursor < end_ms:
        interval_end = min(cursor + window_ms, end_ms)
        intervals.append((f"fixed:{window_ms}:{index}", cursor, interval_end))
        cursor += stride_ms
        index += 1
    return intervals


def _build_candidates(
    intervals: list[tuple[str, int, int]],
    timestamps_ms: list[int],
    embeddings: np.ndarray,
) -> list[VisualCandidate]:
    candidates = []
    for candidate_id, start_ms, end_ms in intervals:
        selected = [
            embedding
            for timestamp_ms, embedding in zip(
                timestamps_ms,
                embeddings,
                strict=True,
            )
            if start_ms <= timestamp_ms < end_ms
        ]
        if not selected:
            continue
        pooled = np.mean(np.stack(selected), axis=0)
        pooled /= max(float(np.linalg.norm(pooled)), 1e-12)
        candidates.append(
            VisualCandidate(candidate_id, start_ms, end_ms, pooled)
        )
    return candidates


def _median_step(timestamps_ms: list[int]) -> int:
    steps = sorted(
        right - left for left, right in zip(timestamps_ms, timestamps_ms[1:])
    )
    return steps[len(steps) // 2]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    app()
