import threading

import networkx as nx


class TemporalIndex:
    """
    Directed temporal graph: edges represent chronological adjacency between segments.
    Enables "N hops forward/backward in time" queries for context-window retrieval.
    """

    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()
        self._lock = threading.Lock()

    def add_segment(self, segment_id: str, prev_segment_id: str | None) -> None:
        with self._lock:
            self._graph.add_node(segment_id)
            if prev_segment_id and self._graph.has_node(prev_segment_id):
                # prev → current (forward edge)
                self._graph.add_edge(prev_segment_id, segment_id, rel="next")
                # current → prev (backward edge)
                self._graph.add_edge(segment_id, prev_segment_id, rel="prev")

    def get_neighbours(self, segment_id: str, hops: int = 2) -> list[str]:
        """
        BFS up to `hops` edges away from segment_id (both directions).
        Returns segment IDs ordered by increasing temporal distance.
        """
        if not self._graph.has_node(segment_id):
            return []

        visited: set[str] = {segment_id}
        queue: list[tuple[str, int]] = [(segment_id, 0)]
        result: list[str] = []

        while queue:
            node, depth = queue.pop(0)
            if depth > 0:
                result.append(node)
            if depth < hops:
                for neighbour in self._graph.neighbors(node):
                    if neighbour not in visited:
                        visited.add(neighbour)
                        queue.append((neighbour, depth + 1))

        return result
