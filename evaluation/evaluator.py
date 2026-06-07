import asyncio
import json
from pathlib import Path

import structlog

from evaluation.metrics import mean_average_precision, mean_ndcg, mrr
from evaluation.qrels_parser import load_qrels, load_topics

logger = structlog.get_logger(__name__)


async def run_evaluation(
    topics_path: Path,
    qrels_path: Path,
    top_k: int = 10,
) -> dict:
    from services.retrieval.searcher import multimodal_search

    topics = load_topics(topics_path)      # {qid: query_text}
    qrels = load_qrels(qrels_path)         # {qid: {relevant_segment_ids}}

    results: dict[str, list[str]] = {}
    for qid, query in topics.items():
        hits = await multimodal_search(text_query=query, audio_url=None, top_k=top_k)
        results[qid] = [h.segment_id for h in hits]

    scores = {
        "MAP": mean_average_precision(qrels, results),
        f"NDCG@{top_k}": mean_ndcg(qrels, results, k=top_k),
        "MRR": mrr(qrels, results),
        "num_queries": len(topics),
    }

    logger.info("evaluation.done", **scores)
    return scores


if __name__ == "__main__":
    import typer

    def main(
        topics: Path = typer.Argument(...),
        qrels: Path = typer.Argument(...),
        top_k: int = 10,
    ):
        scores = asyncio.run(run_evaluation(topics, qrels, top_k))
        print(json.dumps(scores, indent=2))

    typer.run(main)
