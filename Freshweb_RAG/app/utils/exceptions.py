"""Custom exceptions for FreshWeb-RAG."""

from fastapi import HTTPException


class FreshWebException(Exception):
    """Base exception for FreshWeb-RAG."""
    pass


class DocumentNotFoundError(FreshWebException):
    pass


class MemoryNotFoundError(FreshWebException):
    pass


class RetrievalError(FreshWebException):
    pass


class LLMProviderError(FreshWebException):
    pass


class WebSearchError(FreshWebException):
    pass


class EmbeddingError(FreshWebException):
    pass
