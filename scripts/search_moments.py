from __future__ import annotations

import asyncio
import json

import typer

from app.schemas.search import MomentSearchRequest
from services.moment_search import MomentSearchService

app = typer.Typer(help="Run one Moment Search query from the command line.")


@app.command()
def main(
    media_id: str = typer.Option(..., help="Selected media ID to search within."),
    query: str = typer.Option(..., help="Semantic Query."),
    duration_sec: float = typer.Option(..., min=0.001, help="Selected video duration."),
    top_k: int = typer.Option(10, min=1, max=100, help="Top-K Video Moments."),
    profile: str = typer.Option("activitynet_visual_heavy", help="Evaluation Profile."),
) -> None:
    response = asyncio.run(
        MomentSearchService().run(
            MomentSearchRequest(
                media_id=media_id,
                query=query,
                duration_sec=duration_sec,
                top_k=top_k,
                profile=profile,
            )
        )
    )
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
