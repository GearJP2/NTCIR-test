from __future__ import annotations

import typer

from evaluation.index_diagnostics import diagnostics_to_json, inspect_media_index
from storage.milvus.client import get_milvus_client
from storage.milvus.collections import ensure_all_collections

app = typer.Typer(help="Check whether one media ID has indexed Moment Search evidence.")


@app.command()
def main(
    media_id: str = typer.Option(..., help="Media ID to inspect."),
    sample_limit: int = typer.Option(100, min=1, help="Maximum rows sampled per collection."),
) -> None:
    client = get_milvus_client()
    ensure_all_collections(client)
    report = inspect_media_index(
        media_id=media_id,
        milvus_client=client,
        sample_limit=sample_limit,
    )
    print(diagnostics_to_json(report))


if __name__ == "__main__":
    app()
