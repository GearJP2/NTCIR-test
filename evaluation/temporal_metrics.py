from __future__ import annotations

from dataclasses import dataclass

from evaluation.manifest import EvaluationQuery, GroundTruthMoment


@dataclass(frozen=True)
class RetrievedMoment:
    media_id: str
    start_sec: float
    end_sec: float
    score: float
    moment_id: str = ""


def temporal_iou(a: GroundTruthMoment | RetrievedMoment, b: GroundTruthMoment | RetrievedMoment) -> float:
    intersection = max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))
    union = max(a.end_sec, b.end_sec) - min(a.start_sec, b.start_sec)
    if union <= 0.0:
        return 0.0
    return intersection / union


def recall_at_k(
    queries: list[EvaluationQuery],
    results_by_query_id: dict[str, list[RetrievedMoment]],
    k: int = 10,
    tiou_threshold: float = 0.3,
) -> float:
    if not queries:
        return 0.0

    hits = 0
    for query in queries:
        ranked = _ranked(results_by_query_id.get(query.query_id, []))[:k]
        if any(_matches(query, result, tiou_threshold) for result in ranked):
            hits += 1
    return hits / len(queries)


def average_precision_at_k(
    query: EvaluationQuery,
    ranked_results: list[RetrievedMoment],
    k: int = 10,
    tiou_threshold: float = 0.3,
) -> float:
    ranked = _ranked(ranked_results)[:k]
    for rank, result in enumerate(ranked, start=1):
        if _matches(query, result, tiou_threshold):
            return 1.0 / rank
    return 0.0


def mean_average_precision_at_k(
    queries: list[EvaluationQuery],
    results_by_query_id: dict[str, list[RetrievedMoment]],
    k: int = 10,
    tiou_threshold: float = 0.3,
) -> float:
    if not queries:
        return 0.0

    scores = [
        average_precision_at_k(
            query,
            results_by_query_id.get(query.query_id, []),
            k=k,
            tiou_threshold=tiou_threshold,
        )
        for query in queries
    ]
    return sum(scores) / len(scores)


def _matches(query: EvaluationQuery, result: RetrievedMoment, tiou_threshold: float) -> bool:
    if result.media_id != query.media_id:
        return False
    return temporal_iou(query.ground_truth, result) >= tiou_threshold


def _ranked(results: list[RetrievedMoment]) -> list[RetrievedMoment]:
    return sorted(results, key=lambda result: result.score, reverse=True)
