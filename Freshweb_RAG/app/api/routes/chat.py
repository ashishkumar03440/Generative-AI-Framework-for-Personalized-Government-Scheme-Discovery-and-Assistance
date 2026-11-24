"""Chat endpoint — the main RAG entry point with caching, analytics, and persistence."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user_id

logger = logging.getLogger("freshweb_rag")
router = APIRouter()


# ── Request / Response schemas ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    conversation_id: Optional[str] = Field(None)
    use_knowledge_base: bool = Field(True)
    force_web_search: bool = Field(False)
    user_context: Optional[dict] = Field(None)


class CitationItem(BaseModel):
    citation_id: str
    source_type: str
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None
    published_at: Optional[str] = None
    reliability_score: Optional[float] = None
    freshness_score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    route: str
    confidence: float
    citations: list[CitationItem] = []
    sources_used: list[str] = []
    memories_used: list[dict] = []
    limitations: list[str] = []
    verification_status: str
    retrieval_metadata: dict = {}
    execution_time_ms: float
    cache_hit: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────
def _query_cache_key(user_id: str, query: str, use_kb: bool, force_web: bool) -> str:
    raw = f"{user_id}:{query}:{use_kb}:{force_web}"
    return "chat:" + hashlib.sha256(raw.encode()).hexdigest()[:32]


async def _persist_messages(
    db: AsyncSession,
    conversation_id: str,
    user_id: str,
    query: str,
    answer: str,
) -> None:
    """Save user + assistant messages to conversation history."""
    try:
        from app.database.models import ConversationMessage
        from uuid import UUID
        db.add(ConversationMessage(
            conversation_id=UUID(conversation_id),
            user_id=user_id,
            role="user",
            content=query,
        ))
        db.add(ConversationMessage(
            conversation_id=UUID(conversation_id),
            user_id=user_id,
            role="assistant",
            content=answer,
        ))
        await db.commit()
    except Exception as exc:
        logger.warning("Could not persist messages: %s", exc)
        await db.rollback()


async def _ensure_conversation(
    db: AsyncSession,
    user_id: str,
    conversation_id: str,
    query: str,
) -> None:
    """Create a conversation record if it does not already exist."""
    try:
        from app.database.models import Conversation
        from uuid import UUID
        existing = await db.get(Conversation, UUID(conversation_id))
        if not existing:
            title = query[:80] + ("..." if len(query) > 80 else "")
            conv = Conversation(id=UUID(conversation_id), user_id=user_id, title=title)
            db.add(conv)
            await db.commit()
    except Exception as exc:
        logger.warning("Could not ensure conversation record: %s", exc)
        await db.rollback()


# ── Endpoint ──────────────────────────────────────────────────────────────────
@router.post("/chat", response_model=ChatResponse, summary="Adaptive RAG chat")
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    start = time.perf_counter()
    conversation_id = request.conversation_id or str(uuid4())
    cache_hit = False

    # ── Cache check (skip for force_web or freshness-sensitive) ───────────────
    cache_key = _query_cache_key(user_id, request.query, request.use_knowledge_base, request.force_web_search)
    if not request.force_web_search:
        try:
            from app.cache.redis_cache import cache_get
            cached = await cache_get(cache_key)
            if cached:
                cached["conversation_id"] = conversation_id
                cached["cache_hit"] = True
                cached["execution_time_ms"] = (time.perf_counter() - start) * 1000
                return ChatResponse(**cached)
        except Exception:
            pass  # Cache miss is acceptable

    # ── Ensure conversation record exists ─────────────────────────────────────
    await _ensure_conversation(db, user_id, conversation_id, request.query)

    # ── Run RAG graph ─────────────────────────────────────────────────────────
    try:
        from app.graphs.rag_graph import run_rag_graph
        result = await run_rag_graph(
            query=request.query,
            user_id=user_id,
            conversation_id=conversation_id,
            use_knowledge_base=request.use_knowledge_base,
            force_web_search=request.force_web_search,
            db=db,
        )
    except Exception as exc:
        logger.exception("RAG graph error for user=%s", user_id)
        result = {
            "answer": "An error occurred while processing your request. Please try again.",
            "route": "error",
            "confidence": 0.0,
            "citations": [],
            "sources_used": [],
            "memories_used": [],
            "limitations": [str(exc)],
            "verification_status": "error",
            "retrieval_metadata": {},
        }

    elapsed = (time.perf_counter() - start) * 1000

    # ── Build response ────────────────────────────────────────────────────────
    citations = []
    for c in result.get("citations", []):
        try:
            citations.append(CitationItem(**c) if isinstance(c, dict) else c)
        except Exception:
            pass

    response_obj = ChatResponse(
        answer=result.get("answer", ""),
        conversation_id=conversation_id,
        route=result.get("route", "unknown"),
        confidence=result.get("confidence", 0.0),
        citations=citations,
        sources_used=result.get("sources_used", []),
        memories_used=result.get("memories_used", []),
        limitations=result.get("limitations", []),
        verification_status=result.get("verification_status", "unverified"),
        retrieval_metadata=result.get("retrieval_metadata", {}),
        execution_time_ms=elapsed,
        cache_hit=False,
    )

    # ── Persist conversation messages ─────────────────────────────────────────
    await _persist_messages(db, conversation_id, user_id, request.query, response_obj.answer)

    # ── Cache successful non-web responses ────────────────────────────────────
    if response_obj.route not in ("error", "web_forced") and not request.force_web_search:
        try:
            from app.cache.redis_cache import cache_set
            from app.config import settings
            await cache_set(cache_key, response_obj.model_dump(), ttl=settings.cache_ttl_default)
        except Exception:
            pass

    # ── Record analytics (non-blocking) ──────────────────────────────────────
    try:
        from app.analytics.metrics import record_query_analytics
        import asyncio
        asyncio.create_task(
            record_query_analytics(
                user_id=user_id,
                conversation_id=conversation_id,
                query=request.query,
                result=result,
                execution_time_ms=elapsed,
                cache_hit=cache_hit,
            )
        )
    except Exception:
        pass

    return response_obj
