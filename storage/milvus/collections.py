from pymilvus import DataType, MilvusClient

from storage.milvus.schemas import (
    AUDIO_COLLECTION,
    TEXT_COLLECTION,
    VISUAL_COLLECTION,
    audio_schema,
    text_schema,
    visual_schema,
)

_HNSW_INDEX = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}


def ensure_all_collections(client: MilvusClient) -> None:
    # Core search collections
    _ensure(client, AUDIO_COLLECTION, audio_schema(), "audio_vector", 512)
    _ensure(client, VISUAL_COLLECTION, visual_schema(), "visual_vector", 768)
    _ensure(client, TEXT_COLLECTION, text_schema(), "text_vector", 768)

    # Episodic memory collection (csat_episodic_memory) — bootstrapped via its
    # own service class so schema + index logic live in one place.
    from storage.milvus.milvus_service import MilvusService
    MilvusService(client=client)


def _ensure(
    client: MilvusClient,
    name: str,
    schema,
    vec_field: str,
    dim: int,
) -> None:
    if not client.has_collection(name):
        client.create_collection(collection_name=name, schema=schema)
        client.create_index(
            collection_name=name,
            field_name=vec_field,
            index_params=_HNSW_INDEX,
        )
        client.load_collection(name)


def upsert_audio(client: MilvusClient, segments, vectors) -> None:
    data = [
        {
            "segment_id": seg.segment_id,
            "media_id": seg.media_id,
            "start_sec": seg.start_sec,
            "end_sec": seg.end_sec,
            "audio_vector": vec.tolist(),
        }
        for seg, vec in zip(segments, vectors)
    ]
    client.upsert(collection_name=AUDIO_COLLECTION, data=data)


def upsert_text(client: MilvusClient, chunks, vectors) -> None:
    data = [
        {
            "chunk_id": chunk.chunk_id,
            "segment_id": chunk.segment_id,
            "media_id": chunk.media_id,
            "start_sec": chunk.start_sec,
            "end_sec": chunk.end_sec,
            "text": chunk.text,
            "language": chunk.language,
            "text_vector": vec.tolist(),
        }
        for chunk, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=TEXT_COLLECTION, data=data)


def upsert_visual(client: MilvusClient, keyframes, vectors) -> None:
    data = [
        {
            "frame_id": kf.frame_id,
            "media_id": kf.media_id,
            "timestamp_sec": kf.timestamp_sec,
            "visual_vector": vec.tolist(),
        }
        for kf, vec in zip(keyframes, vectors)
    ]
    client.upsert(collection_name=VISUAL_COLLECTION, data=data)
