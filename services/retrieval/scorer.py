def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    id_field: str = "segment_id",
) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) over multiple ranked result lists.

    RRF score for document d:  sum over lists of  1 / (k + rank(d))

    All hit dicts from input lists are merged; the fused score replaces `score`.
    """
    rrf_scores: dict[str, float] = {}
    hit_registry: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            doc_id = hit[id_field]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in hit_registry:
                hit_registry[doc_id] = hit.copy()

    for doc_id, score in rrf_scores.items():
        hit_registry[doc_id]["score"] = score

    return sorted(hit_registry.values(), key=lambda h: h["score"], reverse=True)
