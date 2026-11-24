"""Analytics recording — auto-records query metrics after each chat."""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import QueryAnalyticsRecord
from app.database import session as db_session

logger = logging.getLogger("freshweb_rag")


async def record_query_analytics(
    user_id: str,
    conversation_id: Optional[str],
    query: str,
    result: dict,
    execution_time_ms: float,
    cache_hit: bool = False,
    db: Optional[AsyncSession] = None,
) -> None:
    """
    Persist query analytics to database.

    Called asynchronously after each chat response.
    Opens its own session to avoid transaction race conditions with request lifecycle.
    """
    try:
        retrieval_meta = result.get("retrieval_metadata", {})
        record = QueryAnalyticsRecord(
            user_id=user_id,
            conversation_id=conversation_id,
            query_hash=hashlib.sha256(query.encode()).hexdigest()[:32],
            complexity=retrieval_meta.get("complexity"),
            freshness_required=retrieval_meta.get("freshness_required"),
            memory_intent=retrieval_meta.get("memory_intent"),
            rag_route=result.get("route"),
            retrieval_verdict=retrieval_meta.get("retrieval_verdict"),
            num_chunks_retrieved=retrieval_meta.get("num_kb_chunks", 0),
            num_web_sources=retrieval_meta.get("num_web_sources", 0),
            verification_result=result.get("verification_status"),
            cache_hit=cache_hit,
            execution_time_ms=execution_time_ms,
        )

        async with db_session.AsyncSessionLocal() as session:
            session.add(record)
            await session.commit()

        # MongoDB persistence
        try:
            from app.database.mongodb import mongodb_manager
            await mongodb_manager.insert_analytics({
                "user_id": user_id,
                "conversation_id": conversation_id,
                "query_hash": hashlib.sha256(query.encode()).hexdigest()[:32],
                "complexity": retrieval_meta.get("complexity"),
                "freshness_required": retrieval_meta.get("freshness_required"),
                "memory_intent": retrieval_meta.get("memory_intent"),
                "rag_route": result.get("route"),
                "retrieval_verdict": retrieval_meta.get("retrieval_verdict"),
                "num_chunks_retrieved": retrieval_meta.get("num_kb_chunks", 0),
                "num_web_sources": retrieval_meta.get("num_web_sources", 0),
                "verification_result": result.get("verification_status"),
                "cache_hit": cache_hit,
                "execution_time_ms": execution_time_ms,
            })
        except Exception as m_exc:
            logger.debug("MongoDB analytics record skipped: %s", m_exc)
    except Exception as exc:
        logger.warning("Failed to record analytics (non-critical): %s", exc)
