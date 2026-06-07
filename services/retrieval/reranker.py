import structlog

logger = structlog.get_logger(__name__)


def rerank(query: str, hits: list[dict]) -> list[dict]:
    """
    Cross-encoder reranker: re-scores each hit by running the query + passage
    through a ms-marco cross-encoder and sorting by the new score.
    Falls back to the original ANN order if no passage text is available.
    """
    from model_zoo.registry import ModelRegistry
    model = ModelRegistry.get("reranker")

    pairs = []
    scorable_indices = []

    for idx, hit in enumerate(hits):
        passage = hit.get("text") or hit.get("summary") or ""
        if passage:
            pairs.append((query, passage))
            scorable_indices.append(idx)

    if not pairs:
        logger.debug("reranker.no_text_fallback")
        return hits

    scores = model.predict(pairs)

    for score, idx in zip(scores, scorable_indices):
        hits[idx]["rerank_score"] = float(score)

    # Hits without text keep their original ANN score as rerank_score
    for hit in hits:
        if "rerank_score" not in hit:
            hit["rerank_score"] = hit.get("score", 0.0)

    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
