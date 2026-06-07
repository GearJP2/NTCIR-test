from pymilvus import CollectionSchema, DataType, FieldSchema

AUDIO_COLLECTION = "audio_segments"
VISUAL_COLLECTION = "visual_keyframes"
TEXT_COLLECTION = "text_transcripts"


def audio_schema() -> CollectionSchema:
    return CollectionSchema(
        fields=[
            FieldSchema("segment_id", DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema("media_id", DataType.VARCHAR, max_length=64),
            FieldSchema("start_sec", DataType.FLOAT),
            FieldSchema("end_sec", DataType.FLOAT),
            FieldSchema("summary", DataType.VARCHAR, max_length=512, default_value=""),
            FieldSchema("object_key", DataType.VARCHAR, max_length=512, default_value=""),
            FieldSchema("audio_vector", DataType.FLOAT_VECTOR, dim=512),
        ],
        description="CLAP audio segment embeddings",
        enable_dynamic_field=False,
    )


def visual_schema() -> CollectionSchema:
    return CollectionSchema(
        fields=[
            FieldSchema("frame_id", DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema("media_id", DataType.VARCHAR, max_length=64),
            FieldSchema("timestamp_sec", DataType.FLOAT),
            FieldSchema("visual_vector", DataType.FLOAT_VECTOR, dim=768),
        ],
        description="CLIP keyframe embeddings",
        enable_dynamic_field=False,
    )


def text_schema() -> CollectionSchema:
    return CollectionSchema(
        fields=[
            FieldSchema("chunk_id", DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema("segment_id", DataType.VARCHAR, max_length=64),
            FieldSchema("media_id", DataType.VARCHAR, max_length=64),
            FieldSchema("start_sec", DataType.FLOAT),
            FieldSchema("end_sec", DataType.FLOAT),
            FieldSchema("text", DataType.VARCHAR, max_length=2048, default_value=""),
            FieldSchema("language", DataType.VARCHAR, max_length=8, default_value="th"),
            FieldSchema("object_key", DataType.VARCHAR, max_length=512, default_value=""),
            FieldSchema("text_vector", DataType.FLOAT_VECTOR, dim=768),
        ],
        description="sentence-transformers transcript chunk embeddings",
        enable_dynamic_field=False,
    )
