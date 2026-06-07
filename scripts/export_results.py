"""
Export ranked search results in NTCIR TREC submission format.
Output line format:  qid  Q0  segment_id  rank  score  run_tag
"""

import asyncio
import csv
from pathlib import Path

import typer

from evaluation.qrels_parser import load_topics


def main(
    topics_path: Path = typer.Argument(Path("data/topics/topics.tsv")),
    output_path: Path = typer.Option(Path("results/run.txt")),
    top_k: int = typer.Option(100),
    run_tag: str = typer.Option("NTCIR_CSAT_MM"),
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    topics = load_topics(topics_path)
    asyncio.run(_export(topics, output_path, top_k, run_tag))
    typer.echo(f"Results written to {output_path}")


async def _export(topics: dict, output_path: Path, top_k: int, run_tag: str):
    from services.retrieval.searcher import multimodal_search

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for qid, query in topics.items():
            hits = await multimodal_search(text_query=query, audio_url=None, top_k=top_k)
            for rank, hit in enumerate(hits, start=1):
                writer.writerow([qid, "Q0", hit.segment_id, rank, f"{hit.score:.6f}", run_tag])


if __name__ == "__main__":
    typer.run(main)
