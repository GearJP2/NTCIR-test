"""
AudioService — Core ingestion pipeline for audio files.

Full pipeline (all steps are independently callable):
    audio file
        → chunk  (FIXED_DURATION or VAD)
        → embed  (CLAP via HuggingFace  or  Wav2Vec2)
        → upload chunks to MinIO
        → upsert vectors + metadata into Milvus `csat_episodic_memory`

Public API
----------
service = AudioService(strategy=ChunkingStrategy.VAD, embedder_name="clap")

# Step by step (useful for testing each stage):
chunks = service.process_file(path, media_id)   # segment + embed
chunks = service.upload_chunks(chunks)           # push to MinIO
count  = service.store_in_milvus(chunks)         # upsert to Milvus

# Or one shot:
chunks = service.run_pipeline(path, media_id)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import structlog

from app.core.config import settings
from app.core.exceptions import EmbeddingError, StorageError

logger = structlog.get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_TARGET_SR = 16_000           # All models ultimately expect 16 kHz
_CLAP_SR = 48_000             # CLAP processor requires 48 kHz input
_VAD_FRAME_MS = 30            # webrtcvad: 10 | 20 | 30 ms
_VAD_AGGRESSIVENESS = 2       # 0 (permissive) – 3 (strict)


# ── Data model ────────────────────────────────────────────────────────────────

class ChunkingStrategy(str, Enum):
    FIXED_DURATION = "fixed_duration"   # Split every N seconds with overlap
    VAD = "vad"                         # Voice Activity Detection boundaries


@dataclass
class AudioChunk:
    """One indexed audio segment, fully self-contained for storage."""
    chunk_id: str
    media_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    local_path: str                  # temp WAV file written to disk
    minio_url: str = ""              # filled after upload_chunks()
    object_key: str = ""             # MinIO object key
    embedding: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    transcript: str = ""
    language: str = "th"
    embedding_model: str = "clap"
    created_at: int = field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp())
    )


# ── Main service ──────────────────────────────────────────────────────────────

class AudioService:
    """
    Orchestrates the full audio ingestion pipeline.

    Parameters
    ----------
    strategy:
        `VAD` (default) uses webrtcvad to detect speech boundaries.
        `FIXED_DURATION` splits at fixed intervals with 1-second overlap.
    embedder_name:
        `"clap"` (default) — 512-dim shared audio/text space (HuggingFace CLAP).
        `"wav2vec2"` — 768-dim mean-pooled hidden states.
    chunk_duration_sec:
        Target chunk length used only for FIXED_DURATION strategy.
    min_chunk_sec / max_chunk_sec:
        Guard-rails applied to both strategies.
    """

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.VAD,
        embedder_name: Literal["clap", "wav2vec2"] = "clap",
        chunk_duration_sec: float = 30.0,
        min_chunk_sec: float = 1.0,
        max_chunk_sec: float = 60.0,
    ) -> None:
        self.strategy = strategy
        self.embedder_name = embedder_name
        self.chunk_duration_sec = chunk_duration_sec
        self.min_chunk_sec = min_chunk_sec
        self.max_chunk_sec = max_chunk_sec

    # ── Public pipeline steps ─────────────────────────────────────────────────

    def process_file(
        self,
        audio_path: Path | str,
        media_id: str,
        language: str = "th",
        tmp_dir: Path | None = None,
    ) -> list[AudioChunk]:
        """
        Segment the audio file and embed each chunk.
        Does **not** write to MinIO or Milvus.

        Raises
        ------
        FileNotFoundError   – audio_path does not exist
        EmbeddingError      – model inference failed
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        work_dir = (tmp_dir or audio_path.parent) / "chunks" / media_id
        work_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "audio_service.process.start",
            media_id=media_id,
            strategy=self.strategy,
            embedder=self.embedder_name,
            file=str(audio_path),
        )

        chunks = self._segment(audio_path, media_id, work_dir, language)
        if not chunks:
            logger.warning("audio_service.no_chunks_produced", media_id=media_id)
            return []

        logger.info(
            "audio_service.segment.done", media_id=media_id, n_chunks=len(chunks)
        )

        chunks = self._embed_chunks(chunks)
        logger.info(
            "audio_service.embed.done", media_id=media_id, n_chunks=len(chunks)
        )
        return chunks

    def upload_chunks(self, chunks: list[AudioChunk]) -> list[AudioChunk]:
        """
        Upload each chunk's WAV file to MinIO.
        Fills `minio_url` and `object_key` on each AudioChunk in place.

        Raises
        ------
        StorageError – any MinIO I/O failure
        """
        from storage.minio.operations import get_presigned_url, upload_file

        for chunk in chunks:
            object_key = f"chunks/{chunk.media_id}/{chunk.chunk_id}.wav"
            try:
                upload_file(Path(chunk.local_path), object_key)
                chunk.object_key = object_key
                # 30-day presigned URL; regenerate from object_key when expired
                chunk.minio_url = get_presigned_url(object_key, expires_hours=24 * 30)
                logger.debug(
                    "audio_service.upload.ok",
                    chunk_id=chunk.chunk_id,
                    key=object_key,
                )
            except StorageError:
                raise
            except Exception as exc:
                raise StorageError(
                    f"Failed to upload chunk {chunk.chunk_id} to MinIO: {exc}"
                ) from exc

        logger.info(
            "audio_service.upload.done",
            n_chunks=len(chunks),
            media_id=chunks[0].media_id if chunks else "—",
        )
        return chunks

    def store_in_milvus(self, chunks: list[AudioChunk]) -> int:
        """
        Upsert all chunks into `csat_episodic_memory`.
        Returns the number of rows written.

        Raises
        ------
        StorageError – Milvus upsert failure
        """
        from storage.milvus.milvus_service import MilvusService

        count = MilvusService().upsert_chunks(chunks)
        logger.info(
            "audio_service.milvus_store.done",
            n_written=count,
            media_id=chunks[0].media_id if chunks else "—",
        )
        return count

    def run_pipeline(
        self,
        audio_path: Path | str,
        media_id: str,
        language: str = "th",
        tmp_dir: Path | None = None,
    ) -> list[AudioChunk]:
        """
        Full end-to-end pipeline: process → upload → store.
        Returns fully populated AudioChunk list.
        """
        chunks = self.process_file(audio_path, media_id, language, tmp_dir)
        if not chunks:
            return []
        chunks = self.upload_chunks(chunks)
        self.store_in_milvus(chunks)
        logger.info(
            "audio_service.pipeline.complete",
            media_id=media_id,
            total_chunks=len(chunks),
        )
        return chunks

    # ── Segmentation ──────────────────────────────────────────────────────────

    def _segment(
        self,
        audio_path: Path,
        media_id: str,
        work_dir: Path,
        language: str,
    ) -> list[AudioChunk]:
        audio, sr = _load_audio(audio_path)
        if self.strategy == ChunkingStrategy.FIXED_DURATION:
            return self._segment_fixed(audio, sr, media_id, work_dir, language)
        return self._segment_vad(audio, sr, media_id, work_dir, language)

    def _segment_fixed(
        self,
        audio: np.ndarray,
        sr: int,
        media_id: str,
        work_dir: Path,
        language: str,
    ) -> list[AudioChunk]:
        """
        Slide a window of `chunk_duration_sec` over the audio with a 1-second
        overlap to ensure speech at boundaries is not cut off.
        """
        hop_sec = max(self.chunk_duration_sec - 1.0, 1.0)
        hop_samples = int(hop_sec * sr)
        chunk_samples = int(self.chunk_duration_sec * sr)
        max_samples = int(self.max_chunk_sec * sr)

        chunks: list[AudioChunk] = []
        offset = 0

        while offset < len(audio):
            end = min(offset + chunk_samples, len(audio))
            segment = audio[offset:end]
            duration = len(segment) / sr

            if duration < self.min_chunk_sec:
                break
            if len(segment) > max_samples:
                segment = segment[:max_samples]
                end = offset + max_samples

            chunks.append(
                _save_chunk(
                    segment, sr,
                    start_sec=offset / sr,
                    end_sec=end / sr,
                    media_id=media_id,
                    work_dir=work_dir,
                    language=language,
                )
            )
            offset += hop_samples

        return chunks

    def _segment_vad(
        self,
        audio: np.ndarray,
        sr: int,
        media_id: str,
        work_dir: Path,
        language: str,
    ) -> list[AudioChunk]:
        """
        Segment by speech activity using webrtcvad (30 ms frames, aggressiveness=2).
        Consecutive speech frames are merged until `max_chunk_sec` is reached.
        Segments shorter than `min_chunk_sec` are silently dropped.
        """
        import webrtcvad

        frame_len = int(sr * _VAD_FRAME_MS / 1000)
        frame_bytes = frame_len * 2  # int16 = 2 bytes / sample

        pcm = (audio * 32768).astype(np.int16).tobytes()
        vad = webrtcvad.Vad(_VAD_AGGRESSIVENESS)

        frames = [
            pcm[i: i + frame_bytes]
            for i in range(0, len(pcm) - frame_bytes, frame_bytes)
        ]
        # Drop incomplete trailing frames
        speech_flags: list[bool] = [
            vad.is_speech(f, sr) for f in frames if len(f) == frame_bytes
        ]

        chunks: list[AudioChunk] = []
        seg_start: int | None = None   # frame index of segment start
        seg_end: int = 0               # frame index of last speech frame

        def _flush(start_frame: int, end_frame: int) -> AudioChunk | None:
            start_sec = start_frame * _VAD_FRAME_MS / 1000
            end_sec = (end_frame + 1) * _VAD_FRAME_MS / 1000
            if (end_sec - start_sec) < self.min_chunk_sec:
                return None
            s = int(start_sec * sr)
            e = min(int(end_sec * sr), len(audio))
            return _save_chunk(
                audio[s:e], sr, start_sec, end_sec, media_id, work_dir, language
            )

        for idx, is_speech in enumerate(speech_flags):
            if is_speech:
                if seg_start is None:
                    seg_start = idx
                seg_end = idx

                # Force-flush when max_chunk_sec is exceeded
                accumulated = (idx - seg_start + 1) * _VAD_FRAME_MS / 1000
                if accumulated >= self.max_chunk_sec:
                    chunk = _flush(seg_start, seg_end)
                    if chunk:
                        chunks.append(chunk)
                    seg_start = None
            else:
                if seg_start is not None:
                    chunk = _flush(seg_start, seg_end)
                    if chunk:
                        chunks.append(chunk)
                    seg_start = None

        # Flush any trailing speech
        if seg_start is not None:
            chunk = _flush(seg_start, seg_end)
            if chunk:
                chunks.append(chunk)

        return chunks

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_chunks(self, chunks: list[AudioChunk]) -> list[AudioChunk]:
        embedder = _get_embedder(self.embedder_name)
        try:
            paths = [c.local_path for c in chunks]
            vectors = embedder.embed_batch(paths)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Audio embedding failed: {exc}") from exc

        for chunk, vec in zip(chunks, vectors):
            chunk.embedding = vec
            chunk.embedding_model = self.embedder_name

        return chunks


# ── Audio loader (module-level helper) ───────────────────────────────────────

def _load_audio(path: Path, sr: int = _TARGET_SR) -> tuple[np.ndarray, int]:
    """Load audio file to mono float32 at the target sample rate."""
    import librosa
    audio, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
    return audio.astype(np.float32), loaded_sr


def _save_chunk(
    samples: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    media_id: str,
    work_dir: Path,
    language: str,
) -> AudioChunk:
    chunk_id = str(uuid.uuid4())
    out_path = work_dir / f"{chunk_id}.wav"
    sf.write(str(out_path), samples, sr, subtype="PCM_16")
    return AudioChunk(
        chunk_id=chunk_id,
        media_id=media_id,
        start_sec=round(start_sec, 3),
        end_sec=round(end_sec, 3),
        duration_sec=round(end_sec - start_sec, 3),
        local_path=str(out_path),
        language=language,
    )


def _get_embedder(name: str) -> "BaseAudioEmbedder":
    if name == "clap":
        return ClapAudioEmbedder()
    if name == "wav2vec2":
        return Wav2VecAudioEmbedder()
    raise ValueError(f"Unknown embedder: {name!r}. Choose 'clap' or 'wav2vec2'.")


# ── Embedder base + implementations ──────────────────────────────────────────

class BaseAudioEmbedder:
    """Abstract base for audio file embedders."""

    def embed(self, audio_path: str) -> np.ndarray:
        raise NotImplementedError

    def embed_batch(self, audio_paths: list[str]) -> list[np.ndarray]:
        """Default: sequential. Override for batched GPU inference."""
        return [self.embed(p) for p in audio_paths]


class ClapAudioEmbedder(BaseAudioEmbedder):
    """
    CLAP audio embedder via HuggingFace `transformers`.

    Model : laion/clap-htsat-unfused  (open-source, ~630 MB)
    Output: 512-dim L2-normalised float32 vectors.
    Audio/text share the same embedding space → enables cross-modal search.

    Note: CLAP processor expects 48 kHz input; audio is resampled accordingly.
    """

    MODEL_ID = "laion/clap-htsat-unfused"
    DIM = 512

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device: str = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, ClapModel

        logger.info("clap_embedder.loading", model=self.MODEL_ID)
        cache_dir = settings.model_cache_dir

        self._processor = AutoProcessor.from_pretrained(
            self.MODEL_ID, cache_dir=cache_dir
        )
        self._model = ClapModel.from_pretrained(
            self.MODEL_ID, cache_dir=cache_dir
        )
        self._device = (
            settings.device
            if settings.device != "cuda" or torch.cuda.is_available()
            else "cpu"
        )
        self._model = self._model.to(self._device).eval()
        logger.info("clap_embedder.ready", device=self._device)

    def embed_batch(self, audio_paths: list[str]) -> list[np.ndarray]:
        import torch
        import librosa

        self._load()
        waveforms: list[np.ndarray] = []
        for path in audio_paths:
            # CLAP processor requires 48 kHz
            audio, _ = librosa.load(path, sr=_CLAP_SR, mono=True)
            waveforms.append(audio.astype(np.float32))

        inputs = self._processor(
            audios=waveforms,
            return_tensors="pt",
            sampling_rate=_CLAP_SR,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._model.get_audio_features(**inputs)
            # L2 normalise → unit sphere for cosine similarity
            features = features / features.norm(dim=-1, keepdim=True)

        return [row.cpu().numpy().astype(np.float32) for row in features]

    def embed(self, audio_path: str) -> np.ndarray:
        return self.embed_batch([audio_path])[0]

    def embed_text(self, texts: list[str]) -> list[np.ndarray]:
        """
        Encode text queries into the same CLAP embedding space.
        Used at search time to find audio chunks matching a text query.
        """
        import torch

        self._load()
        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)

        return [row.cpu().numpy().astype(np.float32) for row in features]


class Wav2VecAudioEmbedder(BaseAudioEmbedder):
    """
    Wav2Vec2 audio embedder via HuggingFace `transformers`.

    Model : facebook/wav2vec2-base  (~360 MB)
    Output: 768-dim mean-pooled hidden states, L2-normalised.

    Useful as a drop-in replacement when CLAP is too heavy or when
    language-specific fine-tunes are needed (e.g., Thai ASR models).
    """

    MODEL_ID = "facebook/wav2vec2-base"
    DIM = 768

    def __init__(self) -> None:
        self._model = None
        self._extractor = None
        self._device: str = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        logger.info("wav2vec_embedder.loading", model=self.MODEL_ID)
        cache_dir = settings.model_cache_dir

        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.MODEL_ID, cache_dir=cache_dir
        )
        self._model = Wav2Vec2Model.from_pretrained(
            self.MODEL_ID, cache_dir=cache_dir
        )
        self._device = (
            settings.device
            if settings.device != "cuda" or torch.cuda.is_available()
            else "cpu"
        )
        self._model = self._model.to(self._device).eval()
        logger.info("wav2vec_embedder.ready", device=self._device)

    def embed_batch(self, audio_paths: list[str]) -> list[np.ndarray]:
        import torch
        import librosa

        self._load()
        waveforms: list[np.ndarray] = []
        for path in audio_paths:
            audio, _ = librosa.load(path, sr=_TARGET_SR, mono=True)
            waveforms.append(audio.astype(np.float32))

        inputs = self._extractor(
            waveforms,
            return_tensors="pt",
            sampling_rate=_TARGET_SR,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            hidden = outputs.last_hidden_state  # (batch, time, hidden_dim)

            # Attention-mask–weighted mean pooling
            mask = inputs.get("attention_mask")
            if mask is not None:
                mask_f = mask.unsqueeze(-1).float()
                pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            else:
                pooled = hidden.mean(dim=1)

            # L2 normalise
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)

        return [row.cpu().numpy().astype(np.float32) for row in pooled]

    def embed(self, audio_path: str) -> np.ndarray:
        return self.embed_batch([audio_path])[0]
