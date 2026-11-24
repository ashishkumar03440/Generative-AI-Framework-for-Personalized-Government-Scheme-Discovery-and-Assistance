"""Memory store — saves extracted memories to PostgreSQL and Qdrant."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

logger = logging.getLogger("freshweb_rag")


async def store_memory(
    user_id: str,
    memory_type: str,
    content: str,
    importance_score: float = 0.5,
    confidence_score: float = 0.8,
    source_conversation_id: Optional[str] = None,
    provenance: Optional[str] = None,
    metadata: Optional[dict] = None,
    db: AsyncSession = None,
) -> Optional[str]:
    """
    Persist a memory record to PostgreSQL and index it in Qdrant.

    Returns the memory_id on success, None on failure.
    """
    from app.database.models import MemoryRecord

    memory_id = str(uuid.uuid4())
    point_id = str(uuid.uuid4())

    # 1. Store in PostgreSQL
    if db:
        try:
            record = MemoryRecord(
                id=uuid.UUID(memory_id),
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                importance_score=importance_score,
                confidence_score=confidence_score,
                source_conversation_id=uuid.UUID(source_conversation_id) if source_conversation_id else None,
                provenance=provenance,
                metadata_=metadata or {},
                qdrant_point_id=point_id,
            )
            db.add(record)
            await db.commit()
        except Exception as exc:
            logger.error("Failed to store memory in DB: %s", exc)
            return None

    # 2. Index in Qdrant for semantic retrieval
    try:
        from app.utils.llm_client import embed_texts
        from app.retrieval.static.vector_store import upsert_vectors

        vectors = await embed_texts([content])
        await upsert_vectors(
            collection=settings.qdrant_collection_memory,
            ids=[point_id],
            vectors=vectors,
            payloads=[{
                "memory_id": memory_id,
                "user_id": user_id,
                "memory_type": memory_type,
                "content": content,
                "importance_score": importance_score,
                "source_conversation_id": source_conversation_id,
                "provenance": provenance,
            }],
        )
    except Exception as exc:
        logger.warning("Failed to index memory in Qdrant: %s", exc)

    # 3. Store in MongoDB
    try:
        from app.database.mongodb import mongodb_manager
        await mongodb_manager.insert_memory({
            "id": memory_id,
            "user_id": user_id,
            "memory_type": memory_type,
            "content": content,
            "importance_score": importance_score,
            "confidence_score": confidence_score,
            "source_conversation_id": source_conversation_id,
            "provenance": provenance,
            "metadata": metadata or {},
            "qdrant_point_id": point_id,
        })
    except Exception as m_exc:
        logger.debug("MongoDB store_memory sync skipped: %s", m_exc)

    return memory_id


async def delete_memory_vector(point_id: str) -> None:
    """Remove a memory vector from Qdrant."""
    from app.dependencies import get_qdrant

    client = get_qdrant()
    try:
        await client.delete(
            collection_name=settings.qdrant_collection_memory,
            points_selector=[point_id],
        )
    except Exception as exc:
        logger.warning("Failed to delete memory vector %s: %s", point_id, exc)

