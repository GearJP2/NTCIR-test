import threading

import networkx as nx
import numpy as np


class MemoryStore:
    """
    Thread-safe in-process memory graph backed by NetworkX.
    Each node represents one audio/video segment with metadata + embedding.

    In production, this can be swapped for a Redis-backed graph or
    persisted to disk via pickle / JSON between worker restarts.
    """

    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()
        self._lock = threading.Lock()

    def add_node(
        self,
        segment_id: str,
        media_id: str,
        start_sec: float,
        end_sec: float,
        summary: str,
        embedding: np.ndarray,
    ) -> None:
        with self._lock:
            self._graph.add_node(
                segment_id,
                media_id=media_id,
                start_sec=start_sec,
                end_sec=end_sec,
                summary=summary,
                embedding=embedding,
            )

    def get_node(self, segment_id: str) -> dict:
        data = self._graph.nodes.get(segment_id, {})
        return {"segment_id": segment_id, **data}

    def has_node(self, segment_id: str) -> bool:
        return self._graph.has_node(segment_id)

    def all_segments(self, media_id: str | None = None) -> list[dict]:
        nodes = []
        for sid, data in self._graph.nodes(data=True):
            if media_id is None or data.get("media_id") == media_id:
                nodes.append({"segment_id": sid, **data})
        return nodes

    def __len__(self) -> int:
        return self._graph.number_of_nodes()
