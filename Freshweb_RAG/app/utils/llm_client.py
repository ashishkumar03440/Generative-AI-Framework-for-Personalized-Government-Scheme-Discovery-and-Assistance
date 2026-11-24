"""LLM client factory — supports Gemini and OpenAI-compatible providers."""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config import settings

logger = logging.getLogger("freshweb_rag")


@lru_cache(maxsize=4)
def get_llm_client(provider: str | None = None, model: str | None = None):
    """Return a LangChain LLM instance for the configured provider."""
    _provider = provider or settings.llm_provider
    _model = model or settings.llm_model

    if _provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=_model or "gemini-1.5-flash",
                google_api_key=settings.gemini_api_key,
                temperature=0.0,
            )
        except ImportError:
            logger.warning("langchain-google-genai not installed, falling back to OpenAI")
            _provider = "openai"

    if _provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=_model or "gpt-4o-mini",
            api_key=settings.openai_api_key,
            temperature=0.0,
        )

    raise ValueError(f"Unsupported LLM provider: {_provider}")


class HashEmbeddings:
    """Deterministic, zero-network fallback embedding model (384 dimensions)."""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        import hashlib, math
        vec = [0.0] * self.dimension
        words = text.lower().split()
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            idx = h % self.dimension
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def aembed_query(self, query: str) -> list[float]:
        return self._embed(query)


@lru_cache(maxsize=2)
def get_embedding_client(provider: str | None = None, model: str | None = None):
    """Return an Embeddings instance."""
    _provider = provider or settings.embedding_provider
    _model = model or settings.embedding_model

    if _provider == "google":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model=_model or "models/embedding-001",
                google_api_key=settings.gemini_api_key,
            )
        except Exception as e:
            logger.warning("Google embeddings unavailable (%s), trying local", e)
            _provider = "sentence_transformers"

    if _provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=_model or "text-embedding-3-small",
                api_key=settings.openai_api_key,
            )
        except Exception as e:
            logger.warning("OpenAI embeddings unavailable (%s), trying local", e)
            _provider = "sentence_transformers"

    if _provider in ("sentence_transformers", "local", ""):
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=_model or "all-MiniLM-L6-v2",
                model_kwargs={"local_files_only": True},
            )
        except Exception as e:
            logger.info("Local HuggingFace weights not found (%s). Using fast in-memory HashEmbeddings.", e)
            return HashEmbeddings(dimension=settings.embedding_dimension)

    return HashEmbeddings(dimension=settings.embedding_dimension)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts and return vectors."""
    client = get_embedding_client()
    return await client.aembed_documents(texts)


async def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    client = get_embedding_client()
    return await client.aembed_query(query)

