from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import typer
from PIL import Image

from app.schemas.event import EventKind, EventRecord, ModalityCoverage
from services.dataset.castle_transcripts import (
    TranscriptSpan,
    attach_transcripts_to_events,
)
from services.events.manifest import write_event_manifest
from services.events.transcript_boundaries import (
    contextual_transcript_distances,
    transcript_text_bins,
)
from services.events.visual_boundaries import (
    VisualSample,
    pool_event_embedding,
)
from services.events.visual_segmentation import (
    VisualSegmentationConfig,
    run_visual_segmentation,
)

app = typer.Typer(
    help="Build semantic events from CASTLE frames and optional transcripts."
)


@app.command()
def main(
    frame_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    video_id: str = typer.Option(...),
    participant_id: str = typer.Option(...),
    video_uri: str = typer.Option(...),
    output_manifest: Path = typer.Option(...),
    output_embeddings: Path = typer.Option(...),
    output_scores: Path = typer.Option(...),
    processing_version: str = typer.Option(...),
    model_name: str = typer.Option("ViT-B-32-quickgelu"),
    pretrained: str = typer.Option("openai"),
    batch_size: int = typer.Option(16, min=1),
    detector: str = typer.Option("v1", help="Visual detector: v1 or v2."),
    context_radius: int = typer.Option(2, min=1),
    boundary_percentile: float = typer.Option(85.0, min=0.0, max=100.0),
    min_boundary_score: float = typer.Option(0.01, min=0.0),
    smoothing_radius: int = typer.Option(1, min=0),
    min_event_sec: float = typer.Option(10.0, min=1.0),
    max_event_sec: float = typer.Option(60.0, min=1.0),
    transcript_spans: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Optional cleaned transcript JSONL used as a boundary signal.",
    ),
    transcript_weight: float = typer.Option(0.25, min=0.0, max=1.0),
) -> None:
    paths = sorted(frame_dir.glob("*.jpg"))
    if len(paths) < 2:
        raise typer.BadParameter("at least two sampled frames are required")

    timestamps_ms = [int(path.stem) for path in paths]
    model, preprocess, tokenizer = _load_model(model_name, pretrained)
    embeddings = _encode_images(model, preprocess, paths, batch_size)
    samples = [
        VisualSample(timestamp_ms=timestamp_ms, embedding=embedding)
        for timestamp_ms, embedding in zip(timestamps_ms, embeddings, strict=True)
    ]
    interval_step_ms = _median_step_ms(timestamps_ms)
    experiment_start_ms = timestamps_ms[0]
    experiment_end_ms = timestamps_ms[-1] + interval_step_ms
    selected_spans: list[TranscriptSpan] = []
    transcript_scores = None
    transcript_available = None
    if transcript_spans is not None:
        selected_spans = _load_transcript_spans(transcript_spans, video_id)
        texts = transcript_text_bins(
            timestamps_ms,
            end_ms=experiment_end_ms,
            spans=selected_spans,
        )
        text_embeddings = _encode_texts(model, tokenizer, texts)
        transcript_boundary_scores = contextual_transcript_distances(
            text_embeddings,
            np.asarray([bool(text) for text in texts]),
            context_radius=context_radius,
        )
        transcript_scores = transcript_boundary_scores.scores
        transcript_available = transcript_boundary_scores.available
    segmentation = run_visual_segmentation(
        samples,
        VisualSegmentationConfig(
            name=f"{detector}-semantic-events",
            detector=detector,
            context_radius=context_radius,
            smoothing_radius=smoothing_radius,
            boundary_percentile=boundary_percentile,
            min_boundary_score=min_boundary_score,
            min_event_ms=round(min_event_sec * 1000),
            max_event_ms=round(max_event_sec * 1000),
        ),
        start_ms=experiment_start_ms,
        end_ms=experiment_end_ms,
        transcript_scores=transcript_scores,
        transcript_available=transcript_available,
        transcript_weight=transcript_weight,
    )
    intervals = segmentation.intervals

    signal_name = "VISUAL_TEXT" if transcript_spans is not None else "VISUAL"
    macro_id = f"{video_id}_M_{signal_name}_00001"
    records = [
        EventRecord(
            schema_version="1.0",
            processing_version=processing_version,
            event_id=macro_id,
            participant_id=participant_id,
            video_id=video_id,
            event_kind=EventKind.SEMANTIC_MACRO,
            start_ms=experiment_start_ms,
            end_ms=experiment_end_ms,
            duration_ms=experiment_end_ms - experiment_start_ms,
            boundary_confidence=1.0,
            video_uri=video_uri,
            coverage=ModalityCoverage(video=True),
            raw_evidence_uris={
                "visual_frames": [str(path.resolve()) for path in paths]
            },
        )
    ]
    event_vectors: dict[str, np.ndarray] = {
        macro_id: _normalize(np.mean(embeddings, axis=0))
    }
    for index, interval in enumerate(intervals, start=1):
        event_id = f"{video_id}_E_{signal_name}_{index:05d}"
        records.append(
            EventRecord(
                schema_version="1.0",
                processing_version=processing_version,
                event_id=event_id,
                participant_id=participant_id,
                video_id=video_id,
                event_kind=EventKind.SEMANTIC_MICRO,
                start_ms=interval.start_ms,
                end_ms=interval.end_ms,
                duration_ms=interval.end_ms - interval.start_ms,
                parent_event_id=macro_id,
                boundary_confidence=min(1.0, interval.boundary_confidence),
                video_uri=video_uri,
                coverage=ModalityCoverage(video=True),
                raw_evidence_uris={
                    "visual_frames": [
                        str(path.resolve())
                        for path, timestamp_ms in zip(
                            paths,
                            timestamps_ms,
                            strict=True,
                        )
                        if interval.start_ms <= timestamp_ms < interval.end_ms
                    ]
                },
            )
        )
        event_vectors[event_id] = pool_event_embedding(samples, interval)

    if transcript_spans is not None:
        records = attach_transcripts_to_events(
            records,
            {video_id: selected_spans},
            source_uri_by_video={video_id: str(transcript_spans.resolve())},
        )
    write_event_manifest(output_manifest, records)
    output_embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_embeddings,
        event_ids=np.array(list(event_vectors)),
        embeddings=np.stack(list(event_vectors.values())),
        model_name=np.array(model_name),
        pretrained=np.array(pretrained),
    )
    _write_scores(
        output_scores,
        samples,
        segmentation.visual_scores,
        segmentation.transcript_scores,
        segmentation.transcript_available,
        segmentation.raw_scores,
        segmentation.scores,
        segmentation.boundaries,
    )
    typer.echo(
        f"Wrote {len(intervals)} semantic micro events and one macro event; "
        f"{segmentation.summary.learned_boundary_count} learned boundaries; "
        f"{segmentation.summary.forced_split_count} forced splits."
    )


def encode_frames(
    paths: list[Path],
    *,
    model_name: str,
    pretrained: str,
    batch_size: int,
) -> np.ndarray:
    model, preprocess, _ = _load_model(model_name, pretrained)
    return _encode_images(model, preprocess, paths, batch_size)


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

    batches: list[np.ndarray] = []
    for offset in range(0, len(paths), batch_size):
        batch_paths = paths[offset : offset + batch_size]
        batch = torch.stack(
            [preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
        )
        with torch.no_grad():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        batches.append(features.cpu().numpy().astype(np.float32))
    return np.concatenate(batches)


def _encode_texts(model, tokenizer, texts: list[str]) -> np.ndarray:
    import torch

    tokens = tokenizer([text if text else " " for text in texts])
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)


def _load_transcript_spans(path: Path, video_id: str) -> list[TranscriptSpan]:
    spans = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["video_id"] == video_id:
            spans.append(TranscriptSpan(**row))
    return sorted(spans, key=lambda span: (span.start_ms, span.end_ms))


def _median_step_ms(timestamps_ms: list[int]) -> int:
    steps = np.diff(np.array(timestamps_ms, dtype=np.int64))
    if len(steps) == 0:
        raise ValueError("at least two timestamps are required")
    return int(np.median(steps))


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _write_scores(
    path: Path,
    samples: list[VisualSample],
    visual_scores: np.ndarray,
    transcript_scores: np.ndarray | None,
    transcript_available: np.ndarray | None,
    combined_scores: np.ndarray,
    scores: np.ndarray,
    boundaries,
) -> None:
    boundary_by_timestamp = {
        boundary.timestamp_ms: boundary.score for boundary in boundaries
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp_ms",
                "visual_cosine_distance",
                "transcript_cosine_distance",
                "transcript_available",
                "combined_boundary_score",
                "smoothed_score",
                "selected_boundary",
            ],
        )
        writer.writeheader()
        for index, score in enumerate(scores):
            timestamp_ms = samples[index + 1].timestamp_ms
            writer.writerow(
                {
                    "timestamp_ms": timestamp_ms,
                    "visual_cosine_distance": float(visual_scores[index]),
                    "transcript_cosine_distance": (
                        float(transcript_scores[index])
                        if transcript_scores is not None
                        else ""
                    ),
                    "transcript_available": (
                        bool(transcript_available[index])
                        if transcript_available is not None
                        else False
                    ),
                    "combined_boundary_score": float(combined_scores[index]),
                    "smoothed_score": float(score),
                    "selected_boundary": timestamp_ms in boundary_by_timestamp,
                }
            )


if __name__ == "__main__":
    app()
