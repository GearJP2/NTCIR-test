from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application
    app_env: Literal["development", "staging", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_timeout_sec: float = 30.0

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_bucket: str = "ntcir-media"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Models
    device: str = "cuda"
    whisper_model: str = "large-v3"
    clap_model: str = "laion/larger_clap_general"
    clip_model: str = "ViT-L-14"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    text_encoder_model: str = "sentence-transformers/all-mpnet-base-v2"
    model_cache_dir: str = "./model_cache"

    # LLM reasoning backend
    openai_api_key: str = ""
    llm_provider: str = "openai"      # openai | ollama | none
    llm_model: str = "gpt-4o-mini"    # OpenAI model or Ollama model name
    ollama_url: str = "http://localhost:11434"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.2


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
