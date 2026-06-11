import asyncio
import uuid
from pathlib import Path

import structlog

from app.schemas.media import AudioSegment, MediaAsset, TranscriptChunk, VideoKeyframe
from services.ingestion.audio_processor import segment_audio
from services.ingestion.document_processor import build_transcript_chunks
from services.ingestion.video_processor import extract_audio_track, extract_keyframes

logger = structlog.get_logger(__name__)

ALL_MODALITIES = frozenset({"visual", "audio", "asr"})


async def run_ingestion_pipeline(
    asset: MediaAsset,
    local_path: Path,
    modalities: set[str] | None = None,
    keyframe_interval_sec: float | None = None,
) -> dict:
    """
    Orchestrates the full ingestion pipeline for one media file:
      1. Split into sub-tasks based on content_type
      2. Run ASR on audio segments
      3. Generate embeddings (audio, visual, text)
      4. Update the WorldMM memory store
      5. Upsert vectors into Milvus
    Returns a summary dict with counts of indexed segments.
    """
    selected_modalities = set(modalities or ALL_MODALITIES)
    want_visual = "visual" in selected_modalities
    want_audio = "audio" in selected_modalities
    want_asr = "asr" in selected_modalities
    needs_audio_track = want_audio or want_asr

    is_video = asset.content_type.startswith("video/")
    audio_path = local_path

    if is_video:
        if needs_audio_track:
            logger.info("pipeline.extract_audio", media_id=asset.media_id)
            audio_path = extract_audio_track(local_path)
        if want_visual:
            logger.info("pipeline.extract_keyframes", media_id=asset.media_id)
            keyframes: list[VideoKeyframe] = extract_keyframes(
                local_path,
                asset.media_id,
                interval_sec=keyframe_interval_sec,
            )
        else:
            keyframes = []
    else:
        keyframes = []

    if needs_audio_track:
        logger.info("pipeline.segment_audio", media_id=asset.media_id)
        audio_segments: list[AudioSegment] = await asyncio.to_thread(
            segment_audio, audio_path, asset.media_id
        )
    else:
        audio_segments = []

    # Import here to avoid circular deps at module load time
    from storage.milvus.client import get_milvus_client

    milvus = get_milvus_client()

    chunks: list[TranscriptChunk] = []
    if want_asr and audio_segments:
        from services.asr.whisper_engine import WhisperEngine

        asr = WhisperEngine()
        for seg in audio_segments:
            transcript = await asyncio.to_thread(asr.transcribe, seg.audio_path, asset.language)
            seg_chunks = build_transcript_chunks(transcript, seg, asset.media_id)
            chunks.extend(seg_chunks)

    # Parallel embedding
    audio_vecs = []
    if want_audio and audio_segments:
        from services.embedding.clap_encoder import ClapEncoder

        clap_enc = ClapEncoder()
        audio_vecs = await asyncio.to_thread(
            clap_enc.encode_batch,
            [s.audio_path for s in audio_segments],
        )

    text_vecs = []
    if chunks:
        from services.embedding.text_encoder import TextEncoder

        text_enc = TextEncoder()
        text_vecs = await asyncio.to_thread(text_enc.encode_batch, [c.text for c in chunks])

    visual_vecs = []
    if keyframes:
        from services.embedding.visual_encoder import VisualEncoder

        visual_enc = VisualEncoder()
        visual_vecs = await asyncio.to_thread(
            visual_enc.encode_batch,
            [kf.image_path for kf in keyframes],
        )

    # Memory agent: summarize + build temporal graph
    if audio_vecs:
        from services.memory.memory_agent import MemoryAgent

        memory_agent = MemoryAgent(media_id=asset.media_id)
        for seg, vec in zip(audio_segments, audio_vecs):
            await asyncio.to_thread(memory_agent.perceive, seg, vec)

    # Upsert to Milvus
    from storage.milvus.collections import upsert_audio, upsert_text, upsert_visual
    if audio_vecs:
        upsert_audio(milvus, audio_segments, audio_vecs)
    if text_vecs:
        upsert_text(milvus, chunks, text_vecs)
    if keyframes:
        upsert_visual(milvus, keyframes, visual_vecs)

    logger.info(
        "pipeline.done",
        media_id=asset.media_id,
        audio_segments=len(audio_segments),
        text_chunks=len(chunks),
        keyframes=len(keyframes),
    )
    return {
        "media_id": asset.media_id,
        "audio_segments": len(audio_segments),
        "text_chunks": len(chunks),
        "keyframes": len(keyframes),
    }
