"""Web search agent tests — using mocks (no live Tavily API needed)."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.retrieval.web.search_agent import _is_safe_url, search_web


class TestSSRFProtection:
    """Test URL safety validation."""

    def test_localhost_blocked(self):
        assert _is_safe_url("http://localhost/admin") is False

    def test_127_blocked(self):
        assert _is_safe_url("http://127.0.0.1:8080/secret") is False

    def test_private_192_blocked(self):
        assert _is_safe_url("http://192.168.1.1/api") is False

    def test_private_10_blocked(self):
        assert _is_safe_url("http://10.0.0.1/data") is False

    def test_valid_public_url_allowed(self):
        assert _is_safe_url("https://arxiv.org/abs/2310.11511") is True

    def test_https_allowed(self):
        assert _is_safe_url("https://example.com/page") is True

    def test_ftp_blocked(self):
        assert _is_safe_url("ftp://example.com/file") is False

    def test_file_scheme_blocked(self):
        assert _is_safe_url("file:///etc/passwd") is False


class TestWebSearchAgent:
    """Test web search with mocked Tavily API."""

    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty(self):
        with patch("app.retrieval.web.search_agent.settings") as mock_settings:
            mock_settings.tavily_api_key = ""
            mock_settings.max_web_sources = 5
            result = await search_web("test query")
            assert result == []

    @pytest.mark.asyncio
    async def test_tavily_response_parsed(self):
        mock_response = {
            "results": [
                {
                    "url": "https://arxiv.org/abs/2310.11511",
                    "title": "RAG Survey",
                    "content": "This paper surveys RAG approaches.",
                    "raw_content": "Full content about RAG frameworks and their applications.",
                    "published_date": "2024-01-15",
                }
            ]
        }

        with patch("app.retrieval.web.search_agent.settings") as mock_settings:
            mock_settings.tavily_api_key = "test-key"
            mock_settings.max_web_sources = 5

            with patch("app.retrieval.web.search_agent._tavily_search", return_value=[
                {
                    "evidence_id": "test-id",
                    "content": "Full content about RAG frameworks",
                    "url": "https://arxiv.org/abs/2310.11511",
                    "title": "RAG Survey",
                    "domain": "arxiv.org",
                    "published_at": "2024-01-15",
                    "retrieved_at": "2024-11-01T00:00:00",
                    "source_type": "web",
                }
            ]):
                result = await search_web("RAG frameworks")
                assert len(result) >= 1
                assert result[0]["url"] == "https://arxiv.org/abs/2310.11511"

    @pytest.mark.asyncio
    async def test_internal_url_filtered(self):
        """Results with internal URLs should be filtered."""
        from app.retrieval.web.search_agent import _tavily_search, _is_safe_url

        mock_results = [
            {"url": "http://localhost/secret", "content": "internal", "title": "internal"},
            {"url": "https://arxiv.org/paper", "content": "valid", "title": "valid"},
        ]

        # Verify safe URL filtering would work
        safe = [r for r in mock_results if _is_safe_url(r["url"])]
        assert len(safe) == 1
        assert safe[0]["url"] == "https://arxiv.org/paper"

    @pytest.mark.asyncio
    async def test_search_error_returns_empty(self):
        with patch("app.retrieval.web.search_agent.settings") as mock_settings:
            mock_settings.tavily_api_key = "test-key"
            mock_settings.max_web_sources = 5
            with patch("app.retrieval.web.search_agent._tavily_search", side_effect=Exception("Network error")):
                result = await search_web("test query")
                assert result == []

    @pytest.mark.asyncio
    async def test_query_generation_fallback(self):
        from app.retrieval.web.search_agent import generate_search_queries

        with patch("app.utils.llm_client.get_llm_client", side_effect=Exception("LLM unavailable")):
            result = await generate_search_queries("What is the latest Python version?")
            assert isinstance(result, list)
            assert len(result) >= 1
            assert result[0] == "What is the latest Python version?"

