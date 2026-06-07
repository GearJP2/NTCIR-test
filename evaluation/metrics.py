import math
from collections import defaultdict


def average_precision(relevant: set[str], ranked: list[str]) -> float:
    hits, score = 0, 0.0
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / rank
    return score / max(len(relevant), 1)


def mean_average_precision(qrels: dict[str, set[str]], results: dict[str, list[str]]) -> float:
    aps = [average_precision(qrels.get(qid, set()), results.get(qid, [])) for qid in qrels]
    return sum(aps) / max(len(aps), 1)


def ndcg(relevant: set[str], ranked: list[str], k: int = 10) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def mean_ndcg(qrels: dict[str, set[str]], results: dict[str, list[str]], k: int = 10) -> float:
    scores = [ndcg(qrels.get(qid, set()), results.get(qid, []), k) for qid in qrels]
    return sum(scores) / max(len(scores), 1)


def mrr(qrels: dict[str, set[str]], results: dict[str, list[str]]) -> float:
    total = 0.0
    for qid, ranked in results.items():
        relevant = qrels.get(qid, set())
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in relevant:
                total += 1.0 / rank
                break
    return total / max(len(results), 1)
