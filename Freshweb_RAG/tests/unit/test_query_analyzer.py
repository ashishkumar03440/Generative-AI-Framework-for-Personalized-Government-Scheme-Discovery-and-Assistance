"""Unit tests — Adaptive Query Analyzer."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.router.query_analyzer import _rule_based_analysis, QueryAnalysis


class TestQueryAnalyzerRuleBased:
    """Test rule-based fallback analyzer (no LLM needed)."""

    def test_simple_definition_no_retrieval(self):
        result = _rule_based_analysis("What is machine learning?")
        assert result.complexity == "no_retrieval"
        assert result.freshness_required is False
        assert result.confidence > 0.5

    def test_greeting_no_retrieval(self):
        result = _rule_based_analysis("Hello, how are you?")
        assert result.complexity == "no_retrieval"
        assert result.freshness_required is False

    def test_current_ceo_freshness_required(self):
        result = _rule_based_analysis("Who is the current CEO of OpenAI?")
        assert result.freshness_required is True
        assert result.complexity in ("single_hop", "multi_hop")

    def test_latest_news_freshness(self):
        result = _rule_based_analysis("What are the latest developments in AI?")
        assert result.freshness_required is True

    def test_comparison_multihop(self):
        result = _rule_based_analysis("Compare RAG and fine-tuning approaches for LLMs.")
        assert result.complexity == "multi_hop"

    def test_versus_multihop(self):
        result = _rule_based_analysis("What is the difference between GPT-4 vs Claude?")
        assert result.complexity == "multi_hop"

    def test_stock_price_freshness(self):
        result = _rule_based_analysis("What is the current stock price of Apple?")
        assert result.freshness_required is True

    def test_document_query_single_hop(self):
        result = _rule_based_analysis("What does my uploaded document say about RAG?")
        assert result.complexity in ("single_hop", "no_retrieval")
        assert result.freshness_required is False

    def test_confidence_range(self):
        result = _rule_based_analysis("Explain transformers in NLP.")
        assert 0.0 <= result.confidence <= 1.0

    def test_query_analysis_schema(self):
        result = _rule_based_analysis("Test query")
        assert isinstance(result, QueryAnalysis)
        assert result.complexity in ("no_retrieval", "single_hop", "multi_hop")
        assert isinstance(result.freshness_required, bool)
        assert isinstance(result.sub_questions, list)


class TestQueryAnalyzerLLMFallback:
    """Test that LLM failure falls back to rule-based."""

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self):
        from app.router.query_analyzer import analyze_query

        with patch("app.router.query_analyzer._llm_analyze_query", side_effect=Exception("LLM unavailable")):
            result = await analyze_query("What is the latest version of Python?")
            assert isinstance(result, QueryAnalysis)
            assert result.freshness_required is True  # "latest" triggers freshness

