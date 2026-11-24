"""Unit tests — Memory Extractor."""

from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.memory.memory_extractor import _rule_extract, ExtractedMemory


class TestMemoryExtractor:
    """Test memory extraction logic (no LLM needed for rule-based)."""

    def test_no_messages_returns_empty(self):
        result = _rule_extract([], "conv-001")
        assert result == []

    def test_preference_statement_extracted(self):
        history = [
            {"role": "user", "content": "I prefer using Gemini API for my LLM calls."},
        ]
        result = _rule_extract(history, "conv-001")
        assert len(result) >= 1

    def test_decision_statement_extracted(self):
        history = [
            {"role": "user", "content": "We decided to use Qdrant as the vector database for the project."},
        ]
        result = _rule_extract(history, "conv-001")
        assert len(result) >= 1

    def test_trivial_message_not_extracted(self):
        history = [
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello! How can I help you?"},
        ]
        result = _rule_extract(history, "conv-001")
        assert len(result) == 0

    def test_extracted_memory_has_required_fields(self):
        history = [
            {"role": "user", "content": "My project uses FastAPI and Qdrant for the backend architecture."},
        ]
        result = _rule_extract(history, "conv-001")
        if result:
            mem = result[0]
            assert isinstance(mem, ExtractedMemory)
            assert isinstance(mem.memory_type, str)
            assert isinstance(mem.content, str)
            assert 0.0 <= mem.importance_score <= 1.0

    def test_max_extractions_bounded(self):
        history = [
            {"role": "user", "content": f"I prefer option {i}."} for i in range(20)
        ]
        result = _rule_extract(history, "conv-001")
        assert len(result) <= 5

    def test_provenance_set(self):
        history = [
            {"role": "user", "content": "We decided to use LangGraph for orchestration."}
        ]
        result = _rule_extract(history, "conv-test-001")
        if result:
            assert "conv-test-001" in result[0].provenance

    @pytest.mark.asyncio
    async def test_llm_extract_fallback_on_error(self):
        from app.memory.memory_extractor import extract_memories_from_conversation

        with patch("app.memory.memory_extractor._llm_extract", side_effect=Exception("LLM down")):
            history = [{"role": "user", "content": "My project uses FastAPI."}]
            result = await extract_memories_from_conversation(history, "conv-001")
            assert isinstance(result, list)

