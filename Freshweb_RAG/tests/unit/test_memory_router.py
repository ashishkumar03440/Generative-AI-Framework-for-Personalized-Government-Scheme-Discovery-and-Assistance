"""Unit tests — Memory-Aware Query Router."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.memory_aware_router.router import _rule_analyze, _detect_references


class TestReferenceDetection:
    """Test historical reference detection patterns."""

    def test_no_reference_new_query(self):
        has_ref, ref_type, desc = _detect_references("What is machine learning?")
        assert has_ref is False

    def test_previous_project_reference(self):
        has_ref, ref_type, desc = _detect_references("Continue my previous project.")
        assert has_ref is True
        assert ref_type in ("project", "continuation", "temporal")

    def test_yesterday_reference(self):
        has_ref, ref_type, desc = _detect_references("What model did we discuss yesterday?")
        assert has_ref is True

    def test_continue_where_stopped(self):
        has_ref, ref_type, desc = _detect_references("Continue where we stopped.")
        assert has_ref is True

    def test_same_config_reference(self):
        has_ref, ref_type, desc = _detect_references("Use the same configuration as before.")
        assert has_ref is True

    def test_improve_previous(self):
        has_ref, ref_type, desc = _detect_references("Improve the architecture we designed.")
        assert has_ref is True

    def test_earlier_reference(self):
        has_ref, ref_type, desc = _detect_references("What was our earlier approach?")
        assert has_ref is True


class TestMemoryAwareRouter:
    """Test rule-based memory-aware analysis."""

    def test_new_query_intent(self):
        result = _rule_analyze("What is Python?")
        assert result.intent == "NEW_QUERY"
        assert result.has_historical_reference is False

    def test_continue_project_intent(self):
        result = _rule_analyze("Continue my previous project setup.")
        assert result.intent in ("CONTINUE_PROJECT", "RECALL_INFORMATION", "MODIFY_PREVIOUS_WORK")
        assert result.has_historical_reference is True

    def test_recall_model_intent(self):
        result = _rule_analyze("Which embedding model did we use?")
        assert result.has_historical_reference is True
        assert result.intent in ("RECALL_INFORMATION", "CONTINUE_PROJECT")

    def test_improve_architecture_intent(self):
        result = _rule_analyze("Improve the architecture we designed earlier.")
        assert result.has_historical_reference is True

    def test_reuse_config_intent(self):
        result = _rule_analyze("Use the same model configuration as before.")
        assert result.has_historical_reference is True
        assert result.intent in ("REUSE_PREVIOUS_CONFIGURATION", "RECALL_INFORMATION")

    def test_no_history_no_memory_search(self):
        result = _rule_analyze("Explain transformers in NLP.")
        assert result.intent == "NEW_QUERY"
        assert result.memory_search_query is None

    def test_confidence_range(self):
        result = _rule_analyze("Continue our previous conversation.")
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_memory_aware_fallback(self):
        from app.memory_aware_router.router import analyze_memory_aware

        with patch("app.memory_aware_router.router._llm_analyze", side_effect=Exception("LLM down")):
            result = await analyze_memory_aware(
                query="What was our earlier approach?",
                user_id="user-001",
                db=None,
            )
            assert "intent" in result
            assert "has_historical_reference" in result

    @pytest.mark.asyncio
    async def test_new_query_no_memory_search_needed(self):
        from app.memory_aware_router.router import analyze_memory_aware

        with patch("app.memory_aware_router.router._llm_analyze", side_effect=Exception("LLM down")):
            result = await analyze_memory_aware(
                query="Explain Python decorators.",
                user_id="user-001",
                db=None,
            )
            assert result["intent"] == "NEW_QUERY"
            assert result["has_historical_reference"] is False

    @pytest.mark.asyncio
    async def test_ambiguous_multiple_projects_clarification(self):
        """Ambiguous references at low confidence should indicate clarification needed."""
        from app.memory_aware_router.router import _rule_analyze
        result = _rule_analyze("Continue the project.")
        # Should have historical reference but may need clarification
        assert result.has_historical_reference is True

