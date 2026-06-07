#!/usr/bin/env python3
"""
train_and_index.py — Offline CASTLE2024 Dataset Ingestion Pipeline

Loads the CASTLE-Dataset/CASTLE2024 dataset from HuggingFace, processes every
audio/video/transcript item, extracts dense vector embeddings with CLAP, uploads
raw audio chunks to MinIO, and indexes vectors + metadata into Milvus
`csat_episodic_memory`.

Usage
-----
    # Full run (all splits, default settings):
    python scripts/train_and_index.py

    # Custom split + batch size + embedder:
    python scripts/train_and_index.py \\
        --split train \\
        --batch-size 8 \\
        --embedder clap \\
        --chunk-duration 30 \\
        --max-items 500 \\
        --dry-run

    # Resume an interrupted run (skips already-indexed media IDs):
    python scripts/train_and_index.py --resume

Environment
-----------
Copy .env.example → .env and fill in Milvus / MinIO / HuggingFace credentials.
A HuggingFace token is required for gated datasets:
    export HF_TOKEN=hf_...

Pipeline per dataset item
-------------------------
    HuggingFace item
        → decode audio  (array + sampling_rate  →  16 kHz / 48 kHz WAV)
        → FIXED_DURATION chunk  (default 30 s, 1 s overlap)
        → batch embed            (CLAP audio tower, 512-dim, L2-normalised)
        → upload chunks          (MinIO  castle2024/chunks/{media_id}/*.wav)
        → upsert to Milvus       (csat_episodic_memory, HNSW COSINE)

Progress is checkpointed to `.index_checkpoint.json` so interrupted runs resume
cleanly from the last completed batch.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

# ── Add project root to sys.path so imports work when run as a script ────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import soundfile as sf
import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# ── Project imports (after sys.path patch) ───────────────────────────────────
from app.core.config import settings
from storage.minio.client import ensure_bucket, get_minio_client
from storage.milvus.milvus_service import MilvusService

console = Console()
app = typer.Typer(
    name="train-and-index",
    help="Ingest CASTLE2024 from HuggingFace into Milvus + MinIO.",
    add_completion=False,
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_REPO  = "CASTLE-Dataset/CASTLE2024"
CLAP_MODEL_ID = "laion/clap-htsat-fused"       # fused → better audio quality
CLAP_SR       = 48_000                          # CLAP requires 48 kHz
TARGET_SR     = 16_000                          # fallback / VAD rate
EMBED_DIM     = 512                             # CLAP output dim
MINIO_PREFIX  = "castle2024"                    # bucket key prefix
CHECKPOINT    = Path(".index_checkpoint.json")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """One audio chunk ready for embedding + storage."""
    chunk_id:    str
    media_id:    str
    start_sec:   float
    end_sec:     float
    duration_sec: float
    samples:     np.ndarray      # float32, shape (N,), already at CLAP_SR
    transcript:  str = ""
    language:    str = "en"
    # Filled after embedding:
    embedding:   np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    # Filled after MinIO upload:
    object_key:  str = ""
    minio_url:   str = ""


@dataclass
class RunStats:
    items_seen:    int = 0
    items_ok:      int = 0
    items_skipped: int = 0
    items_errored: int = 0
    chunks_total:  int = 0
    vectors_upserted: int = 0
    start_time:    float = field(default_factory=time.time)

    @property
    def elapsed(self) -> str:
        s = int(time.time() - self.start_time)
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

    @property
    def throughput(self) -> float:
        e = time.time() - self.start_time
        return round(self.items_ok / e, 2) if e > 0 else 0.0


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint() -> set[str]:
    """Return the set of already-indexed media_ids."""
    if CHECKPOINT.exists():
        try:
            return set(json.loads(CHECKPOINT.read_text()))
        except (json.JSONDecodeError, OSError):
            return set()
    return set()


def _save_checkpoint(done: set[str]) -> None:
    CHECKPOINT.write_text(json.dumps(sorted(done)))


# ── Model loader ──────────────────────────────────────────────────────────────

class ClapEmbedder:
    """
    Batch CLAP audio embedder (laion/clap-htsat-fused).

    Manages its own model singleton; safe to instantiate once and reuse
    across all batches in the script.

    Audio input:  float32 waveform at 48 kHz, shape (N,) or (batch, N)
    Output:       float32 L2-normalised vectors, shape (batch, 512)
    """

    def __init__(self, model_id: str = CLAP_MODEL_ID, device: str | None = None) -> None:
        self.model_id = model_id
        self._model    = None
        self._processor = None
        self._device   = device or settings.device

    def load(self) -> None:
        """Explicitly load model weights (called once before the ingestion loop)."""
        import torch
        from transformers import AutoProcessor, ClapModel

        console.print(f"[bold cyan]Loading CLAP model:[/] {self.model_id}")
        cache = settings.model_cache_dir

        self._processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=cache)
        self._model = ClapModel.from_pretrained(self.model_id, cache_dir=cache)

        if self._device == "cuda" and not torch.cuda.is_available():
            console.print("[yellow]CUDA not available — falling back to CPU[/]")
            self._device = "cpu"

        self._model = self._model.to(self._device).eval()
        console.print(
            f"[green]CLAP ready[/] · device={self._device} · dim={EMBED_DIM}"
        )

    def embed_batch(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """
        Embed a batch of audio waveforms.

        Parameters
        ----------
        waveforms:
            List of float32 arrays at CLAP_SR (48 kHz).
            Arrays may have different lengths; the processor pads them.

        Returns
        -------
        np.ndarray
            Shape (batch_size, 512), L2-normalised float32.
        """
        import torch

        if self._model is None:
            raise RuntimeError("Call ClapEmbedder.load() before embed_batch().")

        # Processor expects Python lists of float arrays
        inputs = self._processor(
            audios=[w.tolist() for w in waveforms],
            return_tensors="pt",
            sampling_rate=CLAP_SR,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._model.get_audio_features(**inputs)  # (B, 512)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Encode text queries — useful for later search validation.
        Returns shape (len(texts), 512).
        """
        import torch

        if self._model is None:
            raise RuntimeError("Call ClapEmbedder.load() first.")

        inputs = self._processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().astype(np.float32)


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _load_dataset(split: str, hf_token: str | None, cache_dir: str | None):
    """
    Load CASTLE2024 from HuggingFace.
    Returns the HuggingFace Dataset object.
    """
    from datasets import load_dataset, Audio as HFAudio

    console.print(
        f"[bold cyan]Loading dataset:[/] {DATASET_REPO}  split={split}"
    )
    ds = load_dataset(
        DATASET_REPO,
        split=split,
        token=hf_token or os.getenv("HF_TOKEN"),
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Auto-cast any Audio column to float32 arrays decoded at CLAP_SR
    audio_cols = [
        col for col, feat in ds.features.items()
        if hasattr(feat, "sampling_rate")
    ]
    if audio_cols:
        for col in audio_cols:
            ds = ds.cast_column(col, HFAudio(sampling_rate=CLAP_SR))
        console.print(f"[dim]Audio columns auto-decoded at {CLAP_SR} Hz: {audio_cols}[/]")
    else:
        console.print(
            "[yellow]No Audio columns found — audio will be loaded from file paths.[/]"
        )

    console.print(
        f"[green]Dataset loaded:[/] {len(ds):,} items  ·  "
        f"columns: {ds.column_names}"
    )
    return ds, audio_cols


def _resolve_audio(
    item: dict,
    audio_cols: list[str],
    tmp_dir: Path,
) -> tuple[np.ndarray | None, int, str]:
    """
    Extract a float32 waveform from a dataset item.

    Tries (in order):
    1. A decoded HuggingFace Audio column  ({"array": ..., "sampling_rate": ...})
    2. An 'audio_path' / 'video_path' string pointing to a local file
    3. Loads from an 'audio_bytes' column using soundfile

    Returns (waveform, sampling_rate, source_description)
    """
    # 1. HF Audio feature (already decoded)
    for col in audio_cols:
        audio = item.get(col)
        if isinstance(audio, dict) and "array" in audio:
            arr = np.array(audio["array"], dtype=np.float32)
            sr  = int(audio.get("sampling_rate", CLAP_SR))
            return arr, sr, col

    # 2. File path columns
    for key in ("audio_path", "video_path", "path", "file"):
        path_val = item.get(key)
        if path_val and Path(str(path_val)).exists():
            import librosa
            arr, sr = librosa.load(str(path_val), sr=CLAP_SR, mono=True)
            return arr.astype(np.float32), sr, key

    # 3. Raw bytes column
    for key in ("audio_bytes", "audio_data"):
        raw = item.get(key)
        if raw:
            import soundfile as sf
            arr, sr = sf.read(io.BytesIO(raw if isinstance(raw, bytes) else bytes(raw)))
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            arr = arr.astype(np.float32)
            import librosa
            if sr != CLAP_SR:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=CLAP_SR)
                sr = CLAP_SR
            return arr, sr, key

    return None, 0, "none"


def _resolve_transcript(item: dict) -> str:
    """Extract transcript text from common column names."""
    for key in ("transcript", "text", "asr_text", "caption", "description"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()[:2048]
    return ""


def _resolve_media_id(item: dict, idx: int) -> str:
    """Derive a stable media ID from the item or fall back to index."""
    for key in ("id", "media_id", "video_id", "episode_id", "uid"):
        val = item.get(key)
        if val:
            return str(val)
    return f"castle2024_{idx:06d}"


def _resolve_language(item: dict) -> str:
    for key in ("language", "lang"):
        val = item.get(key)
        if isinstance(val, str):
            return val[:8]
    return "en"


# ── Audio chunker ─────────────────────────────────────────────────────────────

def _chunk_audio(
    waveform: np.ndarray,
    sr: int,
    media_id: str,
    transcript: str,
    language: str,
    chunk_duration_sec: float,
    overlap_sec: float = 1.0,
    min_duration_sec: float = 1.0,
) -> list[Chunk]:
    """
    Fixed-duration chunking with overlap.

    Resamples the waveform to CLAP_SR if needed, then slices into windows
    of `chunk_duration_sec` with a 1-second leading overlap.

    Parameters
    ----------
    overlap_sec:
        Leading overlap between consecutive chunks to catch speech at boundaries.
    min_duration_sec:
        Chunks shorter than this are dropped (prevents zero-padding artifacts).
    """
    import librosa

    # Resample to CLAP_SR if needed
    if sr != CLAP_SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=CLAP_SR)
        sr = CLAP_SR

    hop_samples   = int((chunk_duration_sec - overlap_sec) * sr)
    chunk_samples = int(chunk_duration_sec * sr)
    min_samples   = int(min_duration_sec * sr)

    # Safety: ensure hop is at least 1 second
    hop_samples = max(hop_samples, sr)

    chunks: list[Chunk] = []
    offset = 0

    while offset < len(waveform):
        end = min(offset + chunk_samples, len(waveform))
        segment = waveform[offset:end]

        if len(segment) < min_samples:
            break

        start_sec = offset / sr
        end_sec   = end   / sr

        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            media_id=media_id,
            start_sec=round(start_sec, 3),
            end_sec=round(end_sec, 3),
            duration_sec=round(end_sec - start_sec, 3),
            samples=segment.astype(np.float32),
            transcript=transcript,
            language=language,
        ))
        offset += hop_samples

    return chunks


# ── MinIO uploader ────────────────────────────────────────────────────────────

def _upload_chunks_to_minio(
    chunks: list[Chunk],
    minio_client,
    bucket: str,
) -> list[Chunk]:
    """
    Serialize each chunk's waveform to an in-memory WAV and upload to MinIO.
    Fills chunk.object_key on success; leaves it empty on failure (non-fatal).
    """
    for chunk in chunks:
        object_key = f"{MINIO_PREFIX}/chunks/{chunk.media_id}/{chunk.chunk_id}.wav"
        try:
            buf = io.BytesIO()
            sf.write(buf, chunk.samples, CLAP_SR, format="WAV", subtype="PCM_16")
            buf.seek(0)
            size = buf.getbuffer().nbytes
            minio_client.put_object(
                bucket_name=bucket,
                object_name=object_key,
                data=buf,
                length=size,
                content_type="audio/wav",
            )
            chunk.object_key = object_key
        except Exception as exc:
            # Non-fatal: log and continue so Milvus indexing still happens
            console.print(
                f"[yellow]MinIO upload failed for {chunk.chunk_id}: {exc}[/]"
            )
    return chunks


# ── Main ingestion loop ───────────────────────────────────────────────────────

@app.command()
def main(
    split: str = typer.Option(
        "train",
        help="Dataset split to process (train | test | validation | all).",
    ),
    batch_size: int = typer.Option(
        16,
        help="Number of audio chunks embedded in one GPU batch. Reduce if OOM.",
        min=1, max=512,
    ),
    chunk_duration: float = typer.Option(
        30.0,
        help="Target audio chunk length in seconds.",
        min=5.0, max=300.0,
    ),
    embedder: str = typer.Option(
        "clap",
        help="Embedding model: 'clap' (512-dim) or 'wav2vec2' (768-dim).",
    ),
    max_items: int = typer.Option(
        0,
        help="Process at most N items (0 = all). Useful for smoke tests.",
    ),
    resume: bool = typer.Option(
        False,
        help="Skip already-indexed media IDs (read from .index_checkpoint.json).",
    ),
    dry_run: bool = typer.Option(
        False,
        help="Process and embed but do NOT upload to MinIO or upsert to Milvus.",
    ),
    hf_token: str = typer.Option(
        "",
        envvar="HF_TOKEN",
        help="HuggingFace token for gated datasets.",
    ),
    hf_cache_dir: str = typer.Option(
        "",
        help="Custom HuggingFace dataset cache directory.",
    ),
    milvus_uri: str = typer.Option(
        "",
        help="Override MILVUS_URI from .env.",
    ),
):
    """
    Offline ingestion of CASTLE2024 into Milvus + MinIO.

    Each item is chunked, embedded, uploaded to MinIO, and indexed into
    `csat_episodic_memory`. Progress is checkpointed so you can resume.
    """
    if milvus_uri:
        os.environ["MILVUS_URI"] = milvus_uri

    _print_banner(split, batch_size, chunk_duration, dry_run)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    splits_to_process = (
        ["train", "test", "validation"] if split == "all" else [split]
    )
    stats = RunStats()
    done_ids: set[str] = _load_checkpoint() if resume else set()
    if resume and done_ids:
        console.print(
            f"[dim]Resume mode: skipping {len(done_ids):,} already-indexed items.[/]"
        )

    # ── 2. Load CLAP model ────────────────────────────────────────────────────
    clap = ClapEmbedder(model_id=CLAP_MODEL_ID)
    clap.load()

    # ── 3. Connect to infrastructure ──────────────────────────────────────────
    if not dry_run:
        milvus_svc = MilvusService(embedding_dim=EMBED_DIM)
        minio_client = get_minio_client()
        ensure_bucket()
        console.print(
            f"[green]Infrastructure ready[/]  "
            f"Milvus={settings.milvus_uri}  "
            f"MinIO={settings.minio_endpoint}/{settings.minio_bucket}"
        )
    else:
        milvus_svc  = None
        minio_client = None
        console.print("[yellow]DRY RUN — Milvus + MinIO writes disabled.[/]")

    # ── 4. Main loop over splits ───────────────────────────────────────────────
    for split_name in splits_to_process:
        try:
            ds, audio_cols = _load_dataset(
                split_name,
                hf_token or None,
                hf_cache_dir or None,
            )
        except Exception as exc:
            console.print(f"[red]Failed to load split '{split_name}': {exc}[/]")
            continue

        total = len(ds)
        if max_items > 0:
            total = min(total, max_items)

        console.print(
            f"\n[bold]Processing split:[/] [cyan]{split_name}[/]  "
            f"({total:,} items)"
        )

        with _make_progress() as progress:
            task_items = progress.add_task(
                f"[cyan]{split_name}[/]", total=total
            )
            task_embed = progress.add_task("[magenta]Embed queue[/]", total=None)
            task_store = progress.add_task("[green]Milvus writes[/]", total=None)

            pending_chunks: list[Chunk] = []   # accumulated until batch_size

            def _flush_batch(force: bool = False) -> None:
                nonlocal pending_chunks
                if not pending_chunks:
                    return
                if not force and len(pending_chunks) < batch_size:
                    return

                # ── Embed ─────────────────────────────────────────────────
                try:
                    waveforms = [c.samples for c in pending_chunks]
                    vectors   = clap.embed_batch(waveforms)
                    for chunk, vec in zip(pending_chunks, vectors):
                        chunk.embedding = vec
                    stats.chunks_total += len(pending_chunks)
                    progress.advance(task_embed, len(pending_chunks))
                except Exception as exc:
                    console.print(
                        f"[red]Embed error on batch of "
                        f"{len(pending_chunks)}: {exc}[/]"
                    )
                    pending_chunks = []
                    return

                # ── Upload MinIO ──────────────────────────────────────────
                if not dry_run:
                    pending_chunks = _upload_chunks_to_minio(
                        pending_chunks, minio_client, settings.minio_bucket
                    )

                # ── Upsert Milvus ─────────────────────────────────────────
                if not dry_run:
                    try:
                        # Convert Chunk → AudioChunk-compatible duck type
                        n = milvus_svc.upsert_chunks(
                            [_chunk_to_audio_chunk(c) for c in pending_chunks]
                        )
                        stats.vectors_upserted += n
                        progress.advance(task_store, n)
                    except Exception as exc:
                        console.print(f"[red]Milvus upsert error: {exc}[/]")

                # Free GPU memory: drop waveforms after embedding
                for c in pending_chunks:
                    c.samples = np.array([], dtype=np.float32)

                pending_chunks = []

            # ── Item loop ─────────────────────────────────────────────────
            for idx, item in enumerate(ds):
                if max_items > 0 and stats.items_seen >= max_items:
                    break

                stats.items_seen += 1
                media_id = _resolve_media_id(item, idx)

                # Resume: skip if already done
                if media_id in done_ids:
                    stats.items_skipped += 1
                    progress.advance(task_items)
                    continue

                try:
                    waveform, sr, source = _resolve_audio(
                        item, audio_cols, tmp_dir=Path(tempfile.gettempdir())
                    )
                    if waveform is None or len(waveform) == 0:
                        stats.items_skipped += 1
                        progress.advance(task_items)
                        continue

                    transcript = _resolve_transcript(item)
                    language   = _resolve_language(item)

                    chunks = _chunk_audio(
                        waveform, sr,
                        media_id=media_id,
                        transcript=transcript,
                        language=language,
                        chunk_duration_sec=chunk_duration,
                    )
                    if not chunks:
                        stats.items_skipped += 1
                        progress.advance(task_items)
                        continue

                    pending_chunks.extend(chunks)
                    stats.items_ok += 1

                    # Checkpoint after each successful item
                    done_ids.add(media_id)
                    if stats.items_ok % 50 == 0:
                        _save_checkpoint(done_ids)

                    # Flush when buffer reaches batch_size
                    _flush_batch(force=False)

                except Exception as exc:
                    stats.items_errored += 1
                    console.print(
                        f"\n[red]Error on item {idx} ({media_id}): "
                        f"{type(exc).__name__}: {exc}[/]"
                    )
                    if os.getenv("DEBUG"):
                        traceback.print_exc()

                finally:
                    progress.advance(task_items)
                    progress.update(
                        task_items,
                        description=(
                            f"[cyan]{split_name}[/] "
                            f"ok={stats.items_ok} "
                            f"err={stats.items_errored} "
                            f"vec={stats.vectors_upserted}"
                        ),
                    )

            # Flush any remaining chunks
            _flush_batch(force=True)

        # Checkpoint at end of split
        _save_checkpoint(done_ids)

    # ── 5. Final report ────────────────────────────────────────────────────────
    _print_final_report(stats, dry_run)


# ── Adapter: Chunk → duck-typed AudioChunk for MilvusService ─────────────────

def _chunk_to_audio_chunk(chunk: Chunk):
    """
    Produce an object that satisfies the interface MilvusService.upsert_chunks()
    expects (same field names as services.audio_service.AudioChunk).
    We duck-type it with a simple namespace so we don't import AudioChunk here.
    """
    from types import SimpleNamespace
    from datetime import datetime, timezone

    return SimpleNamespace(
        chunk_id=chunk.chunk_id,
        media_id=chunk.media_id,
        start_sec=chunk.start_sec,
        end_sec=chunk.end_sec,
        duration_sec=chunk.duration_sec,
        minio_url=chunk.minio_url,
        object_key=chunk.object_key,
        transcript=chunk.transcript,
        language=chunk.language,
        embedding_model="clap",
        created_at=int(datetime.now(timezone.utc).timestamp()),
        embedding=chunk.embedding,
    )


# ── Progress bar factory ──────────────────────────────────────────────────────

def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=35),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        expand=True,
    )


# ── Rich output helpers ───────────────────────────────────────────────────────

def _print_banner(split: str, batch_size: int, chunk_dur: float, dry_run: bool) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("[bold]Dataset[/]",     DATASET_REPO)
    table.add_row("[bold]Split[/]",       split)
    table.add_row("[bold]Embed model[/]", CLAP_MODEL_ID)
    table.add_row("[bold]Embed dim[/]",   str(EMBED_DIM))
    table.add_row("[bold]Batch size[/]",  str(batch_size))
    table.add_row("[bold]Chunk dur[/]",   f"{chunk_dur}s")
    table.add_row("[bold]Milvus[/]",      settings.milvus_uri)
    table.add_row("[bold]MinIO[/]",       f"{settings.minio_endpoint}/{settings.minio_bucket}")
    table.add_row("[bold]Dry run[/]",     "[yellow]YES[/]" if dry_run else "[green]NO[/]")

    console.print(
        Panel(table, title="[bold blue]CASTLE2024 Ingestion Pipeline[/]", expand=False)
    )


def _print_final_report(stats: RunStats, dry_run: bool) -> None:
    table = Table(title="[bold green]Ingestion Complete[/]", show_header=True)
    table.add_column("Metric",       style="bold")
    table.add_column("Value",        justify="right")
    table.add_row("Items seen",       f"{stats.items_seen:,}")
    table.add_row("Items processed",  f"[green]{stats.items_ok:,}[/]")
    table.add_row("Items skipped",    f"[dim]{stats.items_skipped:,}[/]")
    table.add_row("Items errored",    f"[red]{stats.items_errored:,}[/]")
    table.add_row("Chunks produced",  f"{stats.chunks_total:,}")
    table.add_row(
        "Vectors upserted",
        f"[{'green' if not dry_run else 'yellow'}]{stats.vectors_upserted:,}[/]"
        + (" (dry run)" if dry_run else ""),
    )
    table.add_row("Throughput",       f"{stats.throughput} items/s")
    table.add_row("Elapsed",          stats.elapsed)

    console.print(table)

    if not dry_run:
        console.print(
            f"\n[dim]Checkpoint saved to:[/] {CHECKPOINT.resolve()}\n"
            f"[dim]Run with [bold]--resume[/] to continue from this point.[/]"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
