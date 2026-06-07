import uuid

from app.schemas.media import AudioSegment, TranscriptChunk


def build_transcript_chunks(
    transcript: dict,
    segment: AudioSegment,
    media_id: str,
    max_words: int = 80,
) -> list[TranscriptChunk]:
    """
    Convert a Whisper transcript dict (with word-level timestamps) into
    fixed-length TranscriptChunk objects suitable for dense embedding.

    `transcript` shape: {"text": str, "words": [{"word": str, "start": float, "end": float}]}
    """
    words = transcript.get("words", [])
    if not words:
        # Fall back to single chunk from full text
        return [
            TranscriptChunk(
                chunk_id=str(uuid.uuid4()),
                media_id=media_id,
                segment_id=segment.segment_id,
                text=transcript.get("text", "").strip(),
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                language=transcript.get("language", "th"),
            )
        ]

    chunks: list[TranscriptChunk] = []
    buf: list[dict] = []

    for word in words:
        buf.append(word)
        if len(buf) >= max_words:
            chunks.append(_make_chunk(buf, segment, media_id, transcript.get("language", "th")))
            buf = []

    if buf:
        chunks.append(_make_chunk(buf, segment, media_id, transcript.get("language", "th")))

    return chunks


def _make_chunk(
    words: list[dict], segment: AudioSegment, media_id: str, language: str
) -> TranscriptChunk:
    text = " ".join(w["word"] for w in words).strip()
    start = segment.start_sec + words[0].get("start", 0.0)
    end = segment.start_sec + words[-1].get("end", 0.0)
    return TranscriptChunk(
        chunk_id=str(uuid.uuid4()),
        media_id=media_id,
        segment_id=segment.segment_id,
        text=text,
        start_sec=start,
        end_sec=end,
        language=language,
    )
