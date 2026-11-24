"""Configuration module using Pydantic BaseSettings.

Provides environment-variable-based configuration for the FreshWeb-RAG backend.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, AnyUrl
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────────────────────
    app_env: str = Field(default="development", validation_alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", validation_alias="APP_HOST")
    app_port: int = Field(default=8000, validation_alias="APP_PORT")
    secret_key: str = Field(default="change-me-in-production", validation_alias="SECRET_KEY")

    # ── LLM Providers ────────────────────────────────────────────────────────
    llm_provider: str = Field(default="gemini", validation_alias="LLM_PROVIDER")
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    llm_model: str = Field(default="gemini-1.5-flash", validation_alias="LLM_MODEL")

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_provider: str = Field(default="google", validation_alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(
        default="models/embedding-001", validation_alias="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=768, validation_alias="EMBEDDING_DIMENSION")

    # ── Database ──────────────────────────────────────────────────────────────
    database_type: str = Field(default="mongodb", validation_alias="DATABASE_TYPE")
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017", validation_alias="MONGODB_URI"
    )
    mongodb_db_name: str = Field(
        default="freshweb_rag", validation_alias="MONGODB_DB_NAME"
    )
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/freshweb",
        validation_alias="DATABASE_URL",
    )

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")
    cache_ttl_default: int = Field(default=300, validation_alias="CACHE_TTL_DEFAULT")
    cache_ttl_web_search: int = Field(default=600, validation_alias="CACHE_TTL_WEB_SEARCH")
    cache_ttl_memory: int = Field(default=120, validation_alias="CACHE_TTL_MEMORY")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    qdrant_collection_documents: str = Field(
        default="documents", validation_alias="QDRANT_COLLECTION_DOCUMENTS"
    )
    qdrant_collection_memory: str = Field(
        default="memory", validation_alias="QDRANT_COLLECTION_MEMORY"
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=1000, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, validation_alias="CHUNK_OVERLAP")
    top_k: int = Field(default=5, validation_alias="TOP_K")
    rerank_top_k: int = Field(default=3, validation_alias="RERANK_TOP_K")

    # ── Web Search ────────────────────────────────────────────────────────────
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")
    max_web_search_iterations: int = Field(
        default=2, validation_alias="MAX_WEB_SEARCH_ITERATIONS"
    )
    max_web_sources: int = Field(default=5, validation_alias="MAX_WEB_SOURCES")

    # ── Memory ────────────────────────────────────────────────────────────────
    memory_enabled: bool = Field(default=True, validation_alias="MEMORY_ENABLED")
    memory_retrieval_top_k: int = Field(
        default=5, validation_alias="MEMORY_RETRIEVAL_TOP_K"
    )
    memory_confidence_threshold: float = Field(
        default=0.75, validation_alias="MEMORY_CONFIDENCE_THRESHOLD"
    )
    memory_importance_threshold: float = Field(
        default=0.5, validation_alias="MEMORY_IMPORTANCE_THRESHOLD"
    )

    # ── Verification ──────────────────────────────────────────────────────────
    max_verification_retries: int = Field(
        default=1, validation_alias="MAX_VERIFICATION_RETRIES"
    )

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_per_minute: int = Field(default=60, validation_alias="RATE_LIMIT_PER_MINUTE")

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: str = Field(default="*", validation_alias="CORS_ORIGINS")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
