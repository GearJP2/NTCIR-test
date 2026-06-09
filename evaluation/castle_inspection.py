from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Protocol

import typer

from app.schemas.search import MomentSearchRequest, MomentSearchResponse
from evaluation.curated_queries import CuratedQuery, load_curated_queries
from services.moment_search import MomentSearchService


class MomentSearcher(Protocol):
    async def run(self, request: MomentSearchRequest) -> MomentSearchResponse:
        ...


async def run_castle_inspection(
    queries_path: Path,
    media_id: str,
    duration_sec: float,
    output_path: Path,
    profile_name: str = "castle_lifelog_balanced",
    top_k: int = 10,
    searcher: MomentSearcher | None = None,
) -> list[dict]:
    queries = load_curated_queries(queries_path)
    service = searcher or MomentSearchService()
    rows = []

    for query in queries:
        effective_media_id = query.media_id or media_id
        response = await service.run(
            MomentSearchRequest(
                media_id=effective_media_id,
                query=query.query,
                top_k=top_k,
                duration_sec=duration_sec,
                profile=profile_name,
            )
        )
        rows.append(_inspection_row(query, response))

    _write_jsonl(rows, output_path)
    return rows


def _inspection_row(query: CuratedQuery, response: MomentSearchResponse) -> dict:
    return {
        "query_id": query.query_id,
        "query": query.query,
        "tags": list(query.tags),
        "notes": query.notes,
        "media_id": response.media_id,
        "profile": response.profile,
        "top_k": response.top_k,
        "results": [
            {
                "rank": moment.rank,
                "moment_id": moment.moment_id,
                "start_sec": moment.start_sec,
                "end_sec": moment.end_sec,
                "score": moment.score,
                "evidence": [
                    {
                        "source_type": evidence.source_type,
                        "score": evidence.score,
                        "source_id": evidence.source_id,
                        "timestamp_sec": evidence.timestamp_sec,
                        "start_sec": evidence.start_sec,
                        "end_sec": evidence.end_sec,
                        "text": evidence.text,
                    }
                    for evidence in moment.evidence
                ],
            }
            for moment in response.results
        ],
    }


def _write_jsonl(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(
    queries_path: Path = typer.Option(
        Path("data/curated_queries/castle_smoke.jsonl"),
        help="CASTLE Curated Query Set JSONL path.",
    ),
    media_id: str = typer.Option(..., help="Default CASTLE media ID to search."),
    duration_sec: float = typer.Option(..., min=0.001, help="Selected video duration."),
    output_path: Path = typer.Option(
        Path("data/inspection/castle_smoke_results.jsonl"),
        help="Manual inspection output JSONL path.",
    ),
    profile_name: str = typer.Option("castle_lifelog_balanced", help="Evaluation Profile."),
    top_k: int = typer.Option(10, min=1, max=100, help="Top-K moments per query."),
) -> None:
    rows = asyncio.run(
        run_castle_inspection(
            queries_path=queries_path,
            media_id=media_id,
            duration_sec=duration_sec,
            output_path=output_path,
            profile_name=profile_name,
            top_k=top_k,
        )
    )
    typer.echo(f"Wrote {len(rows)} inspection rows to {output_path}")


if __name__ == "__main__":
    typer.run(main)
