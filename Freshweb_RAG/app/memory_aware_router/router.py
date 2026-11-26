"""Memory-Aware Query Router (MAQR) — main router module."""

from __future__ import annotations

import logging
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("freshweb_rag")


class MemoryAwareAnalysis(BaseModel):
    intent: Literal[
        "NEW_QUERY",
        "RECALL_INFORMATION",
        "CONTINUE_PROJECT",
        "CONTINUE_CONVERSATION",
        "MODIFY_PREVIOUS_WORK",
        "COMPARE_PREVIOUS_DECISIONS",
        "REUSE_PREVIOUS_CONFIGURATION",
        "CLARIFICATION_REQUIRED",
    ]
    has_historical_reference: bool
    reference_type: Optional[str] = None
    reference_description: Optional[str] = None
    target_entities: List[str] = Field(default_factory=list)
    memory_search_query: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    requires_clarification: bool = False


HISTORICAL_PATTERNS = [
    (r"\b(previous|earlier|last time|before|yesterday|last week)\b", "temporal"),
    (r"\b(my project|our project|the project)\b", "project"),
    (r"\b(we (discussed|decided|chose|built|designed|implemented))\b", "decision"),
    (r"\b(continue|pick up|resume|go back)\b", "continuation"),
    (r"\b(same (model|config|configuration|approach|architecture))\b", "reuse"),
    (r"\b(improve|enhance|modify|update|refactor) (the|our|that)\b", "modification"),
    (r"\b(what|which)\b.*\b(model|architecture|approach|method|configuration|framework)\b.*\bdid we\b", "recall"),
]


def _detect_references(query: str) -> tuple[bool, Optional[str], Optional[str]]:
    """Return (has_ref, ref_type, description)."""
    q = query.lower()
    for pattern, ref_type in HISTORICAL_PATTERNS:
        match = re.search(pattern, q)
        if match:
            return True, ref_type, match.group()
    return False, None, None


async def analyze_memory_aware(
    query: str,
    user_id: str,
    db: AsyncSession,
) -> dict:
    """
    Analyze a query for memory-awareness: detect intent, historical references,
    and determine whether memory retrieval is needed.
    """
    try:
        result = await _llm_analyze(query, user_id, db)
    except Exception as exc:
        logger.warning("LLM memory analysis failed: %s. Using rule-based.", exc)
        result = _rule_analyze(query)

    return result.model_dump()


def _rule_analyze(query: str) -> MemoryAwareAnalysis:
    has_ref, ref_type, ref_desc = _detect_references(query)

    if not has_ref:
        return MemoryAwareAnalysis(
            intent="NEW_QUERY",
            has_historical_reference=False,
            confidence=0.9,
        )

    # Map ref_type to intent
    intent_map = {
        "continuation": "CONTINUE_PROJECT",
        "decision": "RECALL_INFORMATION",
        "temporal": "RECALL_INFORMATION",
        "project": "CONTINUE_PROJECT",
        "reuse": "REUSE_PREVIOUS_CONFIGURATION",
        "modification": "MODIFY_PREVIOUS_WORK",
        "recall": "RECALL_INFORMATION",
    }
    intent = intent_map.get(ref_type, "RECALL_INFORMATION")

    return MemoryAwareAnalysis(
        intent=intent,
        has_historical_reference=True,
        reference_type=ref_type,
        reference_description=ref_desc,
        target_entities=[],
        memory_search_query=query,
        confidence=0.75,
        requires_clarification=False,
    )


async def _llm_analyze(query: str, user_id: str, db: AsyncSession) -> MemoryAwareAnalysis:
    from app.utils.llm_client import get_llm_client
    import json

    llm = get_llm_client()
    prompt = f"""Analyze this user query for memory references and intent.

Query: {query}

Classify the intent:
- NEW_QUERY: No reference to previous conversations or work.
- RECALL_INFORMATION: Asking what was discussed/decided before.
- CONTINUE_PROJECT: Wants to continue working on a previous project.
- CONTINUE_CONVERSATION: Wants to continue a previous conversation thread.
- MODIFY_PREVIOUS_WORK: Wants to improve/change previous work.
- COMPARE_PREVIOUS_DECISIONS: Wants to compare with previous approaches.
- REUSE_PREVIOUS_CONFIGURATION: Wants to use the same settings/config as before.
- CLARIFICATION_REQUIRED: Reference is too ambiguous to resolve.

Respond ONLY with JSON:
{{
  "intent": "<one of the above>",
  "has_historical_reference": true | false,
  "reference_type": "<project|model|architecture|configuration|conversation|code|null>",
  "reference_description": "<what they're referring to or null>",
  "target_entities": ["<entity1>", ...],
  "memory_search_query": "<optimized query to search memories or null>",
  "confidence": <0.0-1.0>,
  "requires_clarification": true | false
}}"""

    from app.utils.llm_client import get_content_text
    response = await llm.ainvoke(prompt)
    text = get_content_text(response)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON in LLM response")
    data = json.loads(match.group())
    return MemoryAwareAnalysis(**data)

