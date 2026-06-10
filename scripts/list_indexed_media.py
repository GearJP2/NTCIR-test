from __future__ import annotations

import json

import typer

from evaluation.index_diagnostics import list_indexed_media_candidates
from storage.milvus.client import get_milvus_client
from storage.milvus.collections import ensure_all_collections

app = typer.Typer(help="List media IDs sampled from Moment Search collections.")


@app.command()
def main(
    sample_limit: int = typer.Option(1000, min=1, help="Rows sampled per collection."),
) -> None:
    client = get_milvus_client()
    ensure_all_collections(client)
    candidates = list_indexed_media_candidates(client, sample_limit=sample_limit)
    print(json.dumps(candidates, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
