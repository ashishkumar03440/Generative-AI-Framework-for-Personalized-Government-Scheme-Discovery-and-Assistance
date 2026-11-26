"""Adaptive Query Analyzer — classifies complexity and freshness."""

from __future__ import annotations

import logging
import re
from typing import List, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger("freshweb_rag")

FRESHNESS_SIGNALS = [
    r"\b(current|latest|today|now|recent|right now|this year|this month|this week|2024|2025|2026)\b",
    r"\b(who is the (ceo|cto|president|prime minister|director))\b",
    r"\b(stock price|share price|exchange rate|weather)\b",
    r"\b(just released|just launched|new version|update)\b",
]

MULTI_HOP_SIGNALS = [
    r"\b(compare|comparison|versus|vs\.?|difference between|similarities)\b",
    r"\b(both|all of|across|multiple|several)\b",
    r"\b(how does .+ relate to|what is the relationship between)\b",
    r"\b(then|after that|following|first .+ then)\b",
]

KB_META_SIGNALS = [
    r"\b(what|which|any|list|show|check|tell)\b.*(document|documents|file|files|pdf|pdfs|data|doc|docs|knowledge base|knowledgebase|uploaded|my file|my files)\b",
    r"\b(do i have|are there|is there|have i uploaded|what do you know about me|what is in my)\b",
    r"\b(what is this document|summarize my document|summarize this file|about my resume|my resume|my cv)\b",
]

NO_RETRIEVAL_SIGNALS = [
    r"^(what is|what are|define|explain|describe) (the )?(concept of |idea of )?[\w\s]+\?*$",
    r"^(hello|hi|hey|thank you|thanks|goodbye|bye)\b",
    r"\b(what does .+ mean|what is meant by)\b",
]


class QueryAnalysis(BaseModel):
    complexity: Literal["no_retrieval", "single_hop", "multi_hop"]
    freshness_required: bool
    freshness_reason: str
    sub_questions: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


def _check_patterns(text: str, patterns: List[str]) -> bool:
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def _rule_based_analysis(query: str) -> QueryAnalysis:
    """Fast rule-based fallback when LLM is unavailable."""
    q = query.strip()

    # Meta-queries about knowledge base or user documents always require retrieval
    if _check_patterns(q, KB_META_SIGNALS):
        return QueryAnalysis(
            complexity="single_hop",
            freshness_required=False,
            freshness_reason="Query asks about uploaded documents or knowledge base content.",
            sub_questions=[],
            confidence=0.95,
        )

    freshness_required = _check_patterns(q, FRESHNESS_SIGNALS)
    freshness_reason = "Query contains time-sensitive keywords." if freshness_required else "No time-sensitive signals detected."

    if _check_patterns(q, NO_RETRIEVAL_SIGNALS) and not freshness_required:
        complexity = "no_retrieval"
        confidence = 0.75
    elif _check_patterns(q, MULTI_HOP_SIGNALS):
        complexity = "multi_hop"
        confidence = 0.80
    else:
        complexity = "single_hop"
        confidence = 0.85

    return QueryAnalysis(
        complexity=complexity,
        freshness_required=freshness_required,
        freshness_reason=freshness_reason,
        sub_questions=[],
        confidence=confidence,
    )


async def analyze_query(query: str) -> QueryAnalysis:
    """
    Analyze query complexity and freshness requirement.

    Attempts LLM-based structured analysis; falls back to rule-based if unavailable.
    """
    # Check KB meta signals first to avoid LLM misclassifying questions about user docs as no_retrieval
    if _check_patterns(query, KB_META_SIGNALS):
        return QueryAnalysis(
            complexity="single_hop",
            freshness_required=False,
            freshness_reason="Query asks about uploaded documents or knowledge base content.",
            sub_questions=[],
            confidence=0.95,
        )

    try:
        return await _llm_analyze_query(query)
    except Exception as exc:
        logger.warning("LLM query analysis failed (%s), using rule-based fallback.", exc)
        return _rule_based_analysis(query)


async def _llm_analyze_query(query: str) -> QueryAnalysis:
    """LLM-based query analysis using structured output."""
    from app.utils.llm_client import get_llm_client

    llm = get_llm_client()

    prompt = f"""Analyze the following user query and classify it.

Query: {query}

Classify along two independent dimensions:

1. COMPLEXITY:
   - "no_retrieval": General knowledge, conversational, or definitional questions that don't need external documents.
   - "single_hop": Questions answerable from a single source or document.
   - "multi_hop": Questions requiring synthesis across multiple sources, comparisons, or multi-step reasoning.

2. FRESHNESS:
   - freshness_required: true if the answer depends on current/recent real-world information (news, prices, current leadership, latest versions, recent events).
   - freshness_required: false for stable knowledge (scientific concepts, history, definitions).

Also decompose complex queries into 2-4 sub-questions if applicable.

Respond ONLY with valid JSON matching this schema:
{{
  "complexity": "no_retrieval" | "single_hop" | "multi_hop",
  "freshness_required": true | false,
  "freshness_reason": "<brief explanation>",
  "sub_questions": ["<sub-question 1>", ...],
  "confidence": <0.0-1.0>
}}"""

    import json
    from app.utils.llm_client import get_content_text
    response = await llm.ainvoke(prompt)
    text = get_content_text(response)

    # Extract JSON from response
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in LLM response")

    data = json.loads(match.group())
    return QueryAnalysis(**data)

