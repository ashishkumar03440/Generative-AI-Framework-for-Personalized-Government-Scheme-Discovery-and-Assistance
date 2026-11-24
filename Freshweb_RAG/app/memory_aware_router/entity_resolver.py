"""Entity resolver — resolves historical references from memory."""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("freshweb_rag")


async def resolve_entity(
    query: str,
    user_id: str,
    reference_type: Optional[str] = None,
    db: AsyncSession = None,
) -> dict:
    """
    Attempt to resolve a historical reference to a concrete entity using memory retrieval.
    """
    from app.memory.memory_retriever import retrieve_memories

    # Build a targeted search query based on reference type
    search_query = query
    if reference_type == "project":
        search_query = f"project {query}"
    elif reference_type == "model":
        search_query = f"model architecture {query}"
    elif reference_type == "configuration":
        search_query = f"configuration settings {query}"

    memories = await retrieve_memories(
        query=search_query,
        user_id=user_id,
        top_k=5,
        db=db,
    )

    if not memories:
        return {
            "resolved": False,
            "entity": None,
            "confidence": 0.0,
            "memories_used": [],
            "message": "No relevant memories found to resolve this reference.",
        }

    # Use top memory as the resolved entity
    top_memory = memories[0]
    confidence = top_memory.get("confidence_score", 0.5)

    return {
        "resolved": confidence >= 0.5,
        "entity": top_memory.get("content"),
        "entity_type": reference_type or top_memory.get("memory_type"),
        "confidence": confidence,
        "memories_used": memories[:3],
        "message": f"Resolved from memory with confidence {confidence:.2f}.",
    }

