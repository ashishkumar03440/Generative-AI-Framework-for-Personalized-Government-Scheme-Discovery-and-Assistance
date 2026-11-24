"""Long-term memory retriever using Qdrant semantic search."""

from __future__ import annotations

import logging
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import settings

logger = logging.getLogger("freshweb_rag")


async def retrieve_memories(
    query: str,
    user_id: str,
    memory_types: Optional[List[str]] = None,
    top_k: int = None,
    db: AsyncSession = None,
) -> List[dict]:
    """
    Retrieve semantically relevant memories for a user.

    Returns list of memory dicts with content, type, importance_score, etc.
    """
    _top_k = top_k or settings.memory_retrieval_top_k

    try:
        return await _semantic_retrieve(query, user_id, memory_types, _top_k)
    except Exception as exc:
        logger.warning("Semantic memory retrieval failed: %s. Falling back to DB.", exc)
        if db:
            return await _db_retrieve(query, user_id, memory_types, _top_k, db)
        return []


async def _semantic_retrieve(
    query: str,
    user_id: str,
    memory_types: Optional[List[str]],
    top_k: int,
) -> List[dict]:
    """Dense vector search over Qdrant memory collection."""
    from app.utils.llm_client import embed_query
    from app.retrieval.static.vector_store import search_vectors

    query_vec = await embed_query(query)
    filter_conds = {"user_id": user_id}

    results = await search_vectors(
        collection=settings.qdrant_collection_memory,
        query_vector=query_vec,
        top_k=top_k,
        filter_conditions=filter_conds,
    )

    memories = []
    for r in results:
        payload = r.get("payload", {})
        if memory_types and payload.get("memory_type") not in memory_types:
            continue
        if r.get("score", 0) < settings.memory_confidence_threshold:
            continue
        memories.append({
            "memory_id": payload.get("memory_id", r.get("id")),
            "memory_type": payload.get("memory_type", "general"),
            "content": payload.get("content", ""),
            "importance_score": payload.get("importance_score", 0.5),
            "confidence_score": r.get("score", 0.5),
            "created_at": payload.get("created_at"),
        })

    return memories


async def _db_retrieve(
    query: str,
    user_id: str,
    memory_types: Optional[List[str]],
    top_k: int,
    db: AsyncSession,
) -> List[dict]:
    """Fallback: retrieve from PostgreSQL sorted by importance."""
    from app.database.models import MemoryRecord

    stmt = (
        select(MemoryRecord)
        .where(MemoryRecord.user_id == user_id, MemoryRecord.is_active == True)
        .order_by(MemoryRecord.importance_score.desc())
        .limit(top_k * 3)
    )
    if memory_types:
        stmt = stmt.where(MemoryRecord.memory_type.in_(memory_types))

    result = await db.execute(stmt)
    records = result.scalars().all()

    # Simple keyword match filter
    query_words = set(query.lower().split())
    scored = []
    for rec in records:
        content_words = set(rec.content.lower().split())
        overlap = len(query_words & content_words) / max(len(query_words), 1)
        scored.append((overlap, rec))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {
            "memory_id": str(rec.id),
            "memory_type": rec.memory_type,
            "content": rec.content,
            "importance_score": rec.importance_score or 0.5,
            "confidence_score": overlap,
            "created_at": rec.created_at.isoformat() if rec.created_at else None,
        }
        for overlap, rec in scored[:top_k]
    ]

