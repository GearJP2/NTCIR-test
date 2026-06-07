import numpy as np
import structlog

from app.schemas.media import AudioSegment
from services.memory.memory_store import MemoryStore
from services.memory.summarizer import summarize_segment
from services.memory.temporal_index import TemporalIndex

logger = structlog.get_logger(__name__)


class MemoryAgent:
    """
    WorldMM-inspired Dynamic Multimodal Memory Agent.

    For each ingested segment the agent:
      1. Perceives the raw audio segment + its embedding vector
      2. Generates a short natural-language summary (via LLM)
      3. Writes a memory node into the MemoryStore graph
      4. Links the node to its temporal neighbours in the TemporalIndex

    This enables temporally-aware retrieval: "what happened N seconds before/after X?"
    """

    def __init__(self, media_id: str):
        self.media_id = media_id
        self._store = MemoryStore()
        self._temporal = TemporalIndex()
        self._prev_segment_id: str | None = None

    def perceive(self, segment: AudioSegment, embedding: np.ndarray) -> None:
        """Process one segment: summarise → store → link temporally."""
        summary = summarize_segment(
            segment_id=segment.segment_id,
            audio_path=segment.audio_path,
        )

        self._store.add_node(
            segment_id=segment.segment_id,
            media_id=self.media_id,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            summary=summary,
            embedding=embedding,
        )

        self._temporal.add_segment(
            segment_id=segment.segment_id,
            prev_segment_id=self._prev_segment_id,
        )
        self._prev_segment_id = segment.segment_id

        logger.debug(
            "memory.agent.perceived",
            media_id=self.media_id,
            segment_id=segment.segment_id,
            summary_preview=summary[:80],
        )

    def get_context_window(self, segment_id: str, window: int = 2) -> list[dict]:
        """Return `window` segments before and after the given segment_id."""
        neighbours = self._temporal.get_neighbours(segment_id, hops=window)
        return [self._store.get_node(sid) for sid in neighbours if self._store.has_node(sid)]
