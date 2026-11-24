"""Query rewriter — enriches queries with memory context."""

from __future__ import annotations

import logging
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("freshweb_rag")


async def enrich_with_memory(
    query: str,
    user_id: str,
    conversation_id: Optional[str] = None,
    db: AsyncSession = None,
) -> dict:
    """
    Enrich a query with relevant memory context.

    Returns the original query, enriched query, and memories used.
    """
    from app.memory.memory_retriever import retrieve_memories

    memories = await retrieve_memories(
        query=query,
        user_id=user_id,
        top_k=3,
        db=db,
    )

    if not memories:
        return {
            "original_query": query,
            "enriched_query": query,
            "memories_used": [],
            "enrichment_applied": False,
        }

    # Build memory context string
    memory_context = "\n".join(
        f"[Memory {i+1}] ({m.get('memory_type', 'general')}): {m.get('content', '')}"
        for i, m in enumerate(memories)
    )

    # Try LLM-based enrichment
    try:
        enriched = await _llm_enrich(query, memory_context)
    except Exception as exc:
        logger.warning("LLM enrichment failed: %s", exc)
        # Simple concatenation fallback
        enriched = f"{query}\n\nContext from previous sessions:\n{memory_context}"

    return {
        "original_query": query,
        "enriched_query": enriched,
        "memories_used": memories,
        "enrichment_applied": True,
    }


async def _llm_enrich(query: str, memory_context: str) -> str:
    from app.utils.llm_client import get_llm_client

    llm = get_llm_client()
    prompt = f"""Rewrite the following user query to be more specific and actionable by incorporating the relevant memory context.

Original Query: {query}

Relevant Memory Context:
{memory_context}

Rules:
1. Preserve the user's original intent completely.
2. Add only genuinely relevant context from memory.
3. Keep the rewritten query concise (under 500 words).
4. Never invent information not present in the memory context.
5. If the memory context is irrelevant, return the original query unchanged.

Rewritten Query:"""

    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    return text.strip()

