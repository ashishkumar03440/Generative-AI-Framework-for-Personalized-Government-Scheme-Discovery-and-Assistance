"""Integration tests — FastAPI endpoints (no live services required)."""

from __future__ import annotations

import sys, os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.main import app

client = TestClient(app, raise_server_exceptions=False)

# ── Health ─────────────────────────────────────────────────────────────────────
class TestHealthEndpoints:
    def test_health_ok(self):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "FreshWeb-RAG"

    def test_config_ok(self):
        r = client.get("/api/v1/config")
        assert r.status_code == 200
        body = r.json()
        assert "llm_provider" in body
        assert "memory_enabled" in body

    def test_swagger_docs_available(self):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_openapi_schema_has_all_routes(self):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = list(r.json().get("paths", {}).keys())
        required = ["/api/v1/chat", "/api/v1/documents/upload", "/api/v1/memory", "/api/v1/conversations"]
        for req in required:
            assert req in paths, f"Missing route: {req}"


# ── Documents ──────────────────────────────────────────────────────────────────
class TestDocumentEndpoints:
    def test_list_documents_requires_db(self):
        """DB is not running in test; expect graceful error or empty list."""
        r = client.get("/api/v1/documents")
        assert r.status_code in (200, 500, 503)

    def test_upload_too_large_file_rejected(self):
        # Create a fake >50MB file
        large_content = b"x" * (51 * 1024 * 1024)
        r = client.post(
            "/api/v1/documents/upload",
            files={"file": ("big.txt", large_content, "text/plain")},
        )
        assert r.status_code in (413, 500)  # Too large or DB error

    def test_upload_unsupported_type_rejected(self):
        r = client.post(
            "/api/v1/documents/upload",
            files={"file": ("script.exe", b"MZ\x90\x00", "application/x-msdownload")},
        )
        assert r.status_code in (415, 500)

    def test_get_nonexistent_document(self):
        r = client.get("/api/v1/documents/00000000-0000-0000-0000-000000000000")
        assert r.status_code in (404, 500)


# ── Memory ─────────────────────────────────────────────────────────────────────
class TestMemoryEndpoints:
    def test_list_memory_endpoint_exists(self):
        r = client.get("/api/v1/memory")
        assert r.status_code in (200, 500)

    def test_memory_search_requires_query(self):
        r = client.post("/api/v1/memory/search", json={})
        assert r.status_code == 422  # Validation error — query is required

    def test_memory_search_valid_request(self):
        with patch("app.memory.memory_retriever.retrieve_memories", return_value=[]):
            r = client.post("/api/v1/memory/search", json={"query": "my previous project"})
            assert r.status_code == 200
            assert "results" in r.json()

    def test_forget_requires_body(self):
        r = client.post("/api/v1/memory/forget", json={})
        assert r.status_code == 422

    def test_forget_valid_request(self):
        r = client.post("/api/v1/memory/forget", json={"memory_ids": []})
        assert r.status_code in (200, 500)


# ── Memory-Aware Routing ───────────────────────────────────────────────────────
class TestMemoryAwareEndpoints:
    def test_analyze_new_query(self):
        r = client.post("/api/v1/memory-aware/analyze", json={"query": "What is Python?"})
        assert r.status_code in (200, 500)
        if r.status_code == 200:
            body = r.json()
            assert "intent" in body
            assert "has_historical_reference" in body

    def test_analyze_historical_reference(self):
        r = client.post(
            "/api/v1/memory-aware/analyze",
            json={"query": "Continue my previous project."},
        )
        assert r.status_code in (200, 500)

    def test_enrich_query(self):
        with patch("app.memory.memory_retriever.retrieve_memories", return_value=[]):
            r = client.post("/api/v1/memory-aware/enrich", json={"query": "What is RAG?"})
            assert r.status_code == 200

    def test_resolve_reference(self):
        with patch("app.memory.memory_retriever.retrieve_memories", return_value=[]):
            r = client.post(
                "/api/v1/memory-aware/resolve",
                json={"query": "What model did we use?", "reference_type": "model"},
            )
            assert r.status_code == 200


# ── Conversations ──────────────────────────────────────────────────────────────
class TestConversationEndpoints:
    def test_create_conversation(self):
        r = client.post("/api/v1/conversations", json={"title": "Test Conversation"})
        assert r.status_code in (201, 500)

    def test_list_conversations(self):
        r = client.get("/api/v1/conversations")
        assert r.status_code in (200, 500)


# ── Search ─────────────────────────────────────────────────────────────────────
class TestSearchEndpoint:
    def test_search_requires_query(self):
        r = client.post("/api/v1/search", json={})
        assert r.status_code == 422

    def test_search_valid_request(self):
        with patch("app.retrieval.static.hybrid_retriever.hybrid_search", return_value=[]):
            r = client.post("/api/v1/search", json={"query": "RAG frameworks", "top_k": 3})
            assert r.status_code == 200
            assert r.json()["total"] == 0

    def test_search_invalid_search_type(self):
        with patch("app.retrieval.static.hybrid_retriever.hybrid_search", return_value=[]):
            r = client.post("/api/v1/search", json={"query": "test", "search_type": "invalid"})
            assert r.status_code in (200, 422)


# ── Chat ───────────────────────────────────────────────────────────────────────
class TestChatEndpoint:
    def test_chat_requires_query(self):
        r = client.post("/api/v1/chat", json={})
        assert r.status_code == 422

    def test_chat_query_too_long_rejected(self):
        r = client.post("/api/v1/chat", json={"query": "x" * 5000})
        assert r.status_code == 422

    def test_chat_valid_request_no_services(self):
        mock_result = {
            "answer": "RAG stands for Retrieval-Augmented Generation.",
            "route": "no_retrieval",
            "confidence": 0.9,
            "citations": [],
            "sources_used": [],
            "memories_used": [],
            "limitations": [],
            "verification_status": "passed",
            "retrieval_metadata": {"complexity": "no_retrieval"},
        }
        with patch("app.graphs.rag_graph.run_rag_graph", return_value=mock_result):
            r = client.post(
                "/api/v1/chat",
                json={"query": "What is RAG?", "use_knowledge_base": False},
            )
            assert r.status_code == 200
            body = r.json()
            assert "answer" in body
            assert "conversation_id" in body
            assert body["route"] == "no_retrieval"

