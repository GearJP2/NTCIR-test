import asyncio
import uuid
from pathlib import Path

import structlog

from app.schemas.media import AudioSegment, MediaAsset, TranscriptChunk, VideoKeyframe
from services.ingestion.audio_processor import segment_audio
from services.ingestion.document_processor import build_transcript_chunks
from services.ingestion.video_processor import extract_audio_track, extract_keyframes

logger = structlog.get_logger(__name__)


async def run_ingestion_pipeline(asset: MediaAsset, local_path: Path) -> dict:
    """
    Orchestrates the full ingestion pipeline for one media file:
      1. Split into sub-tasks based on content_type
      2. Run ASR on audio segments
      3. Generate embeddings (audio, visual, text)
      4. Update the WorldMM memory store
      5. Upsert vectors into Milvus
    Returns a summary dict with counts of indexed segments.
    """
    is_video = asset.content_type.startswith("video/")
    audio_path = local_path

    if is_video:
        logger.info("pipeline.extract_audio", media_id=asset.media_id)
        audio_path = await asyncio.to_thread(extract_audio_track, local_path)
        logger.info("pipeline.extract_keyframes", media_id=asset.media_id)
        keyframes: list[VideoKeyframe] = await asyncio.to_thread(
            extract_keyframes, local_path, asset.media_id
        )
    else:
        keyframes = []

    logger.info("pipeline.segment_audio", media_id=asset.media_id)
    audio_segments: list[AudioSegment] = await asyncio.to_thread(
        segment_audio, audio_path, asset.media_id
    )

    # Import here to avoid circular deps at module load time
    from services.asr.whisper_engine import WhisperEngine
    from services.embedding.clap_encoder import ClapEncoder
    from services.embedding.fusion import fuse_embeddings
    from services.embedding.text_encoder import TextEncoder
    from services.embedding.visual_encoder import VisualEncoder
    from services.memory.memory_agent import MemoryAgent
    from storage.milvus.client import get_milvus_client

    asr = WhisperEngine()
    text_enc = TextEncoder()
    clap_enc = ClapEncoder()
    visual_enc = VisualEncoder()
    memory_agent = MemoryAgent(media_id=asset.media_id)
    milvus = get_milvus_client()

    chunks: list[TranscriptChunk] = []
    for seg in audio_segments:
        transcript = await asyncio.to_thread(asr.transcribe, seg.audio_path, asset.language)
        seg_chunks = build_transcript_chunks(transcript, seg, asset.media_id)
        chunks.extend(seg_chunks)

    # Parallel embedding
    audio_vecs, text_vecs = await asyncio.gather(
        asyncio.to_thread(clap_enc.encode_batch, [s.audio_path for s in audio_segments]),
        asyncio.to_thread(text_enc.encode_batch, [c.text for c in chunks]),
    )
    visual_vecs = (
        await asyncio.to_thread(visual_enc.encode_batch, [kf.image_path for kf in keyframes])
        if keyframes else []
    )

    # Memory agent: summarize + build temporal graph
    for seg, vec in zip(audio_segments, audio_vecs):
        await asyncio.to_thread(memory_agent.perceive, seg, vec)

    # Upsert to Milvus
    from storage.milvus.collections import upsert_audio, upsert_text, upsert_visual
    upsert_audio(milvus, audio_segments, audio_vecs)
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
