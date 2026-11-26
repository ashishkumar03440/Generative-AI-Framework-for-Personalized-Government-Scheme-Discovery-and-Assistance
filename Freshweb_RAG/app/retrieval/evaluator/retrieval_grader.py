"""CRAG-style retrieval evaluator."""

from __future__ import annotations

import logging
import re
from typing import List, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger("freshweb_rag")


class RetrievalGrade(BaseModel):
    verdict: Literal["correct", "ambiguous", "incorrect"]
    confidence: float = Field(ge=0.0, le=1.0)
    missing_aspects: List[str] = Field(default_factory=list)
    reasoning: str


async def grade_retrieval(
    query: str,
    retrieved_chunks: List[dict],
) -> RetrievalGrade:
    """
    Evaluate whether retrieved chunks adequately address the query.

    Routing:
        CORRECT   → proceed with KB evidence
        AMBIGUOUS → keep KB evidence + supplement with web search
        INCORRECT → discard KB evidence, use web search only
    """
    if not retrieved_chunks:
        return RetrievalGrade(
            verdict="incorrect",
            confidence=0.95,
            missing_aspects=["No documents retrieved"],
            reasoning="Knowledge base returned no results for this query.",
        )

    # For meta-queries about the user's documents or knowledge base, automatically validate
    q_lower = query.lower()
    if any(k in q_lower for k in ["document", "documents", "file", "files", "knowledge base", "knowledgebase", "uploaded", "resume", "cv", "pdf"]):
        return RetrievalGrade(
            verdict="correct",
            confidence=0.95,
            missing_aspects=[],
            reasoning="Retrieved chunks belong to the user's knowledge base documents.",
        )

    try:
        return await _llm_grade(query, retrieved_chunks)
    except Exception as exc:
        logger.warning("LLM grading failed: %s. Using heuristic.", exc)
        return _heuristic_grade(query, retrieved_chunks)


def _heuristic_grade(query: str, chunks: List[dict]) -> RetrievalGrade:
    """Simple heuristic: check keyword overlap between query and chunks."""
    import re

    # If query is asking about knowledge base / uploaded documents and we have chunks, verdict is correct!
    q_lower = query.lower()
    if any(k in q_lower for k in ["document", "documents", "file", "files", "knowledge base", "knowledgebase", "uploaded", "resume", "cv"]):
        if chunks:
            return RetrievalGrade(
                verdict="correct",
                confidence=0.9,
                missing_aspects=[],
                reasoning="Retrieved chunks correspond to user's knowledge base documents.",
            )

    def tokens(text: str) -> set:
        return set(re.findall(r"\w+", text.lower()))

    query_tokens = tokens(query)
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "of", "to", "and", "or"}
    query_tokens -= stop_words

    combined = " ".join(c.get("content", "") for c in chunks[:3])
    doc_tokens = tokens(combined)

    if not query_tokens:
        overlap = 0.5
    else:
        overlap = len(query_tokens & doc_tokens) / len(query_tokens)

    if overlap > 0.4:
        return RetrievalGrade(
            verdict="correct",
            confidence=min(0.6 + overlap * 0.3, 0.9),
            missing_aspects=[],
            reasoning=f"Keyword overlap {overlap:.0%} suggests retrieved documents are relevant.",
        )
    elif overlap >= 0.15:
        return RetrievalGrade(
            verdict="ambiguous",
            confidence=0.6,
            missing_aspects=["Partial keyword match; may need supplemental sources"],
            reasoning=f"Keyword overlap {overlap:.0%} indicates partial relevance.",
        )
    else:
        return RetrievalGrade(
            verdict="incorrect",
            confidence=0.75,
            missing_aspects=["Low keyword overlap", "Retrieved documents seem off-topic"],
            reasoning=f"Keyword overlap {overlap:.0%} suggests irrelevant retrieval.",
        )


async def _llm_grade(query: str, chunks: List[dict]) -> RetrievalGrade:
    """LLM-based grading with structured output."""
    from app.utils.llm_client import get_llm_client
    import json

    llm = get_llm_client()
    excerpts = "\n---\n".join(
        f"[Chunk {i+1}] {c.get('content', '')[:400]}"
        for i, c in enumerate(chunks[:3])
    )

    prompt = f"""You are a retrieval quality evaluator. Determine if the retrieved documents adequately answer the query.

Query: {query}

Retrieved Documents:
{excerpts}

Evaluate relevance, coverage, and completeness. Respond ONLY with JSON:
{{
  "verdict": "correct" | "ambiguous" | "incorrect",
  "confidence": <0.0-1.0>,
  "missing_aspects": ["aspect1", ...],
  "reasoning": "<brief explanation>"
}}

Rules:
- "correct": Documents clearly address all aspects of the query.
- "ambiguous": Documents partially address the query or may be outdated.
- "incorrect": Documents are irrelevant, off-topic, or completely missing information.
- Do NOT use semantic similarity alone to decide "correct". Check actual content coverage."""

    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in LLM grading response")
    data = json.loads(match.group())
    return RetrievalGrade(**data)

