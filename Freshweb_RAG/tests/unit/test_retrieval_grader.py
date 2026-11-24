"""Unit tests — CRAG Retrieval Evaluator."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.retrieval.evaluator.retrieval_grader import _heuristic_grade, RetrievalGrade


class TestRetrievalGraderHeuristic:
    """Test heuristic retrieval grading (no LLM needed)."""

    async def test_empty_chunks_incorrect(self):
        from app.retrieval.evaluator.retrieval_grader import grade_retrieval
        result = await grade_retrieval("test query", [])
        assert result.verdict == "incorrect"
        assert result.confidence > 0.8

    def test_high_overlap_correct(self):
        query = "What is retrieval augmented generation in NLP?"
        chunks = [{"content": "Retrieval augmented generation (RAG) is a technique in NLP that combines retrieval with generation."}]
        result = _heuristic_grade(query, chunks)
        assert result.verdict == "correct"
        assert result.confidence > 0.5

    def test_low_overlap_incorrect(self):
        query = "Who won the football championship?"
        chunks = [{"content": "Python is a programming language used for machine learning and data science."}]
        result = _heuristic_grade(query, chunks)
        assert result.verdict == "incorrect"

    def test_partial_overlap_ambiguous(self):
        query = "What is the latest RAG framework performance?"
        chunks = [{"content": "RAG frameworks are used for question answering tasks."}]
        result = _heuristic_grade(query, chunks)
        assert result.verdict in ("ambiguous", "correct")

    def test_grade_has_reasoning(self):
        query = "test"
        chunks = [{"content": "test content here"}]
        result = _heuristic_grade(query, chunks)
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    def test_grade_confidence_range(self):
        query = "test query for confidence"
        chunks = [{"content": "test query matching content for confidence check"}]
        result = _heuristic_grade(query, chunks)
        assert 0.0 <= result.confidence <= 1.0

    def test_verdict_literal_values(self):
        query = "some query"
        chunks = [{"content": "some content"}]
        result = _heuristic_grade(query, chunks)
        assert result.verdict in ("correct", "ambiguous", "incorrect")

    def test_multiple_chunks_aggregated(self):
        query = "What is machine learning and deep learning?"
        chunks = [
            {"content": "Machine learning is a subset of AI."},
            {"content": "Deep learning uses neural networks."},
        ]
        result = _heuristic_grade(query, chunks)
        # Multiple matching chunks should trend toward correct
        assert result.verdict in ("correct", "ambiguous")

    @pytest.mark.asyncio
    async def test_llm_grade_fallback_on_error(self):
        from app.retrieval.evaluator.retrieval_grader import grade_retrieval

        with patch("app.retrieval.evaluator.retrieval_grader._llm_grade", side_effect=Exception("LLM down")):
            result = await grade_retrieval("test query", [{"content": "test content"}])
            assert isinstance(result, RetrievalGrade)
            assert result.verdict in ("correct", "ambiguous", "incorrect")

