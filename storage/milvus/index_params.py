"""Helpers for pymilvus IndexParams (v2.4+ API)."""

from pymilvus.milvus_client.index import IndexParams


def hnsw_cosine_index(
    field_name: str,
    *,
    m: int = 16,
    ef_construction: int = 200,
) -> IndexParams:
    params = IndexParams()
    params.add_index(
        field_name=field_name,
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": m, "efConstruction": ef_construction},
    )
    return params
