"""Shared pytest fixtures for FreshWeb-RAG tests."""

from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def app():
    """Create the FastAPI test application."""
    from app.main import app as fastapi_app
    return fastapi_app


@pytest.fixture(scope="session")
def client(app):
    """Return a synchronous TestClient for the app."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def mock_db():
    """Mock SQLAlchemy async session."""
    db = AsyncMock()
    db.get = AsyncMock(return_value=None)
    db.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))))
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.refresh = AsyncMock()
    db.close = AsyncMock()
    db.delete = AsyncMock()
    return db


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock()
    redis.delete = AsyncMock()
    return redis


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client."""
    qdrant = AsyncMock()
    qdrant.search = AsyncMock(return_value=[])
    qdrant.upsert = AsyncMock()
    qdrant.delete = AsyncMock()
    qdrant.collection_exists = AsyncMock(return_value=True)
    return qdrant


@pytest.fixture
def mock_llm():
    """Mock LLM client returning a fixed response."""
    llm = AsyncMock()
    response = MagicMock()
    response.content = "This is a test answer. [1]"
    llm.ainvoke = AsyncMock(return_value=response)
    return llm


@pytest.fixture
def sample_chunks():
    """Sample KB evidence chunks."""
    return [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "content": "RAG (Retrieval Augmented Generation) combines retrieval and generation.",
            "score": 0.85,
            "source": "rag_intro.pdf",
            "page_number": 1,
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "doc-001",
            "content": "Vector databases like Qdrant store dense embeddings for semantic search.",
            "score": 0.78,
            "source": "rag_intro.pdf",
            "page_number": 2,
        },
    ]


@pytest.fixture
def sample_web_evidence():
    """Sample web evidence items."""
    return [
        {
            "evidence_id": "web-001",
            "content": "The latest RAG frameworks use hybrid retrieval combining dense and sparse methods.",
            "url": "https://example.com/rag-2024",
            "title": "Modern RAG Frameworks 2024",
            "domain": "example.com",
            "published_at": "2024-10-01",
            "retrieved_at": "2024-11-01T00:00:00",
            "source_type": "web",
        }
    ]


@pytest.fixture
def sample_memories():
    """Sample memory records."""
    return [
        {
            "memory_id": "mem-001",
            "memory_type": "semantic",
            "content": "User prefers Gemini API for LLM calls in their FreshWeb-RAG project.",
            "importance_score": 0.85,
            "confidence_score": 0.90,
            "created_at": "2024-10-01T00:00:00",
        }
    ]

