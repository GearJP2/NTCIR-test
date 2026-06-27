from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from app.schemas.event import EventKind, EventRecord, ModalityCoverage, RecordingRecord
from scripts.build_visual_semantic_events import (
    _encode_images,
    _encode_texts,
    _load_model,
    _load_transcript_spans,
    _median_step_ms,
    _normalize,
    _write_scores,
)
from scripts.sample_castle_frames import sample_remote_frames, sample_timestamps
from services.dataset.castle_transcripts import attach_transcripts_to_events
from services.events.manifest import write_event_manifest
from services.events.transcript_boundaries import (
    contextual_transcript_distances,
    transcript_text_bins,
)
from services.events.visual_boundaries import VisualSample, pool_event_embedding
from services.events.visual_segmentation import (
    VisualSegmentationConfig,
    run_visual_segmentation,
)

app = typer.Typer(help="Process one CASTLE recording into semantic events for LANTA.")


@app.command()
def main(
    recordings_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    task_id: int = typer.Option(..., min=1, help="1-based SLURM array task id."),
    output_root: Path = typer.Option(Path("processed/lanta_semantic")),
    frame_interval_sec: float = typer.Option(5.0, min=1.0),
    processing_version: str = typer.Option("castle-lanta-semantic-v1"),
    model_name: str = typer.Option("ViT-B-32-quickgelu"),
    pretrained: str = typer.Option("openai"),
    device: str = typer.Option("auto"),
    batch_size: int = typer.Option(64, min=1),
    detector: str = typer.Option("v2"),
    context_radius: int = typer.Option(2, min=1),
    boundary_percentile: float = typer.Option(85.0, min=0.0, max=100.0),
    min_boundary_score: float = typer.Option(0.01, min=0.0),
    smoothing_radius: int = typer.Option(1, min=0),
    min_event_sec: float = typer.Option(10.0, min=1.0),
    max_event_sec: float = typer.Option(60.0, min=1.0),
    transcript_spans: Path | None = typer.Option(None, exists=True, dir_okay=False),
    transcript_weight: float = typer.Option(0.25, min=0.0, max=1.0),
) -> None:
    recordings = _load_recordings(recordings_path)
    if task_id > len(recordings):
        raise typer.BadParameter(
            f"task_id {task_id} exceeds recording count {len(recordings)}"
        )
    recording = recordings[task_id - 1]
    frame_root = output_root / "frames"
    video_dir = frame_root / recording.video_id
    manifest_path = output_root / "manifests" / f"{recording.video_id}.jsonl"
    embeddings_path = output_root / "embeddings" / f"{recording.video_id}.npz"
    scores_path = output_root / "scores" / f"{recording.video_id}.csv"
    done_path = output_root / "done" / f"{recording.video_id}.json"
    if done_path.exists() and manifest_path.exists() and embeddings_path.exists():
        typer.echo(f"Skipping completed {recording.video_id}")
        return

    timestamps = sample_timestamps(recording.duration_ms, frame_interval_sec)
    sample_remote_frames(recording, timestamps, frame_root)
    frame_paths = sorted(video_dir.glob("*.jpg"))
    if len(frame_paths) < 2:
        raise ValueError(f"{recording.video_id} requires at least two frames")

    records, event_vectors, score_payload = _build_semantic_records(
        recording=recording,
        frame_paths=frame_paths,
        transcript_spans=transcript_spans,
        processing_version=processing_version,
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        batch_size=batch_size,
        detector=detector,
        context_radius=context_radius,
        boundary_percentile=boundary_percentile,
        min_boundary_score=min_boundary_score,
        smoothing_radius=smoothing_radius,
        min_event_sec=min_event_sec,
        max_event_sec=max_event_sec,
        transcript_weight=transcript_weight,
    )
    write_event_manifest(manifest_path, records)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        embeddings_path,
        event_ids=np.array(list(event_vectors)),
        embeddings=np.stack(list(event_vectors.values())),
        model_name=np.array(model_name),
        pretrained=np.array(pretrained),
    )
    _write_scores(scores_path, **score_payload)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text(
        json.dumps(
            {
                "video_id": recording.video_id,
                "manifest_path": str(manifest_path),
                "embeddings_path": str(embeddings_path),
                "scores_path": str(scores_path),
                "event_count": len(records),
                "frame_count": len(frame_paths),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"Completed {recording.video_id}: {len(records)} records")


def _build_semantic_records(
    *,
    recording: RecordingRecord,
    frame_paths: list[Path],
    transcript_spans: Path | None,
    processing_version: str,
    model_name: str,
    pretrained: str,
    device: str,
    batch_size: int,
    detector: str,
    context_radius: int,
    boundary_percentile: float,
    min_boundary_score: float,
    smoothing_radius: int,
    min_event_sec: float,
    max_event_sec: float,
    transcript_weight: float,
) -> tuple[list[EventRecord], dict[str, np.ndarray], dict]:
    timestamps_ms = [int(path.stem) for path in frame_paths]
    model, preprocess, tokenizer = _load_model(model_name, pretrained, device=device)
    embeddings = _encode_images(model, preprocess, frame_paths, batch_size)
    samples = [
        VisualSample(timestamp_ms=timestamp_ms, embedding=embedding)
        for timestamp_ms, embedding in zip(timestamps_ms, embeddings, strict=True)
    ]
    step_ms = _median_step_ms(timestamps_ms)
    start_ms = timestamps_ms[0]
    end_ms = min(recording.duration_ms, timestamps_ms[-1] + step_ms)

    selected_spans = []
    transcript_scores = None
    transcript_available = None
    if transcript_spans is not None:
        selected_spans = _load_transcript_spans(transcript_spans, recording.video_id)
        texts = transcript_text_bins(timestamps_ms, end_ms=end_ms, spans=selected_spans)
        text_embeddings = _encode_texts(model, tokenizer, texts)
        transcript = contextual_transcript_distances(
            text_embeddings,
            np.asarray([bool(text) for text in texts]),
            context_radius=context_radius,
        )
        transcript_scores = transcript.scores
        transcript_available = transcript.available

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
        start_ms=start_ms,
        end_ms=end_ms,
        transcript_scores=transcript_scores,
        transcript_available=transcript_available,
        transcript_weight=transcript_weight,
    )
    signal_name = "VISUAL_TEXT" if transcript_spans is not None else "VISUAL"
    macro_id = f"{recording.video_id}_M_{signal_name}_00001"
    records = [
        EventRecord(
            schema_version="1.0",
            processing_version=processing_version,
            event_id=macro_id,
            participant_id=recording.participant_id,
            video_id=recording.video_id,
            event_kind=EventKind.SEMANTIC_MACRO,
            start_ms=start_ms,
            end_ms=end_ms,
            duration_ms=end_ms - start_ms,
            boundary_confidence=1.0,
            video_uri=recording.video_uri,
            coverage=ModalityCoverage(video=True),
            raw_evidence_uris={"visual_frames": [str(path) for path in frame_paths]},
        )
    ]
    event_vectors: dict[str, np.ndarray] = {
        macro_id: _normalize(np.mean(embeddings, axis=0))
    }
    for index, interval in enumerate(segmentation.intervals, start=1):
        event_id = f"{recording.video_id}_E_{signal_name}_{index:05d}"
        records.append(
            EventRecord(
                schema_version="1.0",
                processing_version=processing_version,
                event_id=event_id,
                participant_id=recording.participant_id,
                video_id=recording.video_id,
                event_kind=EventKind.SEMANTIC_MICRO,
                start_ms=interval.start_ms,
                end_ms=interval.end_ms,
                duration_ms=interval.end_ms - interval.start_ms,
                parent_event_id=macro_id,
                boundary_confidence=min(1.0, interval.boundary_confidence),
                video_uri=recording.video_uri,
                coverage=ModalityCoverage(video=True),
                raw_evidence_uris={
                    "visual_frames": [
                        str(path)
                        for path, timestamp_ms in zip(
                            frame_paths,
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
            {recording.video_id: selected_spans},
            source_uri_by_video={recording.video_id: str(transcript_spans)},
        )
    return (
        records,
        event_vectors,
        {
            "samples": samples,
            "visual_scores": segmentation.visual_scores,
            "transcript_scores": segmentation.transcript_scores,
            "transcript_available": segmentation.transcript_available,
            "combined_scores": segmentation.raw_scores,
            "scores": segmentation.scores,
            "boundaries": segmentation.boundaries,
        },
    )


def _load_recordings(path: Path) -> list[RecordingRecord]:
    return [
        RecordingRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    app()
