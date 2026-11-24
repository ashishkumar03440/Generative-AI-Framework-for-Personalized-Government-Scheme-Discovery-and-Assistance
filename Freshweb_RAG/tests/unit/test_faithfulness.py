"""Unit tests — Faithfulness Verification."""

from __future__ import annotations

import pytest
from unittest.mock import patch

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.verification.faithfulness_checker import _heuristic_verify


class TestFaithfulnessChecker:
    """Test heuristic faithfulness verification (no LLM needed)."""

    def test_well_cited_answer_passes(self):
        answer = "RAG combines retrieval and generation [1]. It uses dense embeddings for search [2]."
        evidence = [
            {"content": "RAG (Retrieval Augmented Generation) combines retrieval with generation using dense embeddings."},
            {"content": "Dense embedding search is used in modern RAG pipelines for semantic similarity."},
        ]
        result = _heuristic_verify(answer, evidence)
        assert result.status in ("passed", "partial")

    def test_unsupported_claims_flagged(self):
        answer = "The model was invented in 1842 by Charles Babbage and uses quantum processors."
        evidence = [{"content": "Modern machine learning models use GPU processors for training."}]
        result = _heuristic_verify(answer, evidence)
        assert len(result.unsupported_claims) > 0

    async def test_empty_answer_skipped(self):
        from app.verification.faithfulness_checker import verify_answer

        result = await verify_answer("", [{"content": "some evidence"}])
        assert result.status == "skipped"

    async def test_empty_evidence_skipped(self):
        from app.verification.faithfulness_checker import verify_answer

        result = await verify_answer("some answer", [])
        assert result.status == "skipped"

    def test_invalid_citation_number_flagged(self):
        answer = "This is supported by evidence [99]."
        evidence = [{"content": "one evidence item"}]  # Only 1 item, [99] is invalid
        result = _heuristic_verify(answer, evidence)
        assert len(result.invalid_citations) > 0

    def test_verification_result_has_all_fields(self):
        answer = "Test answer [1]."
        evidence = [{"content": "test content for the answer"}]
        result = _heuristic_verify(answer, evidence)
        assert hasattr(result, "status")
        assert hasattr(result, "supported_claims")
        assert hasattr(result, "unsupported_claims")
        assert hasattr(result, "invalid_citations")
        assert hasattr(result, "corrective_actions")

    def test_to_dict_complete(self):
        answer = "Test [1]."
        evidence = [{"content": "test evidence"}]
        result = _heuristic_verify(answer, evidence)
        d = result.to_dict()
        assert "status" in d
        assert "supported_claims" in d
        assert "unsupported_claims" in d

    @pytest.mark.asyncio
    async def test_llm_verify_fallback_on_error(self):
        from app.verification.faithfulness_checker import verify_answer

        with patch("app.verification.faithfulness_checker._llm_verify", side_effect=Exception("LLM down")):
            result = await verify_answer("Test answer [1].", [{"content": "test evidence here"}])
            assert result.status in ("passed", "partial", "failed")

