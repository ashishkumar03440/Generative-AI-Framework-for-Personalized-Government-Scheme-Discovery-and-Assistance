"""Memory-Aware Query Router (MAQR) endpoints."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user_id

logger = logging.getLogger("freshweb_rag")
router = APIRouter()


class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None


class ResolveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    reference_type: Optional[str] = None
    reference_description: Optional[str] = None


class EnrichRequest(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None


@router.post("/memory-aware/analyze", summary="Analyze query for memory references")
async def analyze_query(
    body: AnalyzeRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    try:
        from app.memory_aware_router.router import analyze_memory_aware
        result = await analyze_memory_aware(query=body.query, user_id=user_id, db=db)
        return result
    except Exception as exc:
        logger.exception("Memory-aware analysis error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/memory-aware/resolve", summary="Resolve a historical reference from memory")
async def resolve_reference(
    body: ResolveRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    try:
        from app.memory_aware_router.entity_resolver import resolve_entity
        result = await resolve_entity(
            query=body.query,
            user_id=user_id,
            reference_type=body.reference_type,
            db=db,
        )
        return result
    except Exception as exc:
        logger.exception("Entity resolution error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/memory-aware/enrich", summary="Enrich query with historical memory context")
async def enrich_query(
    body: EnrichRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    try:
        from app.memory_aware_router.query_rewriter import enrich_with_memory
        result = await enrich_with_memory(
            query=body.query,
            user_id=user_id,
            conversation_id=body.conversation_id,
            db=db,
        )
        return result
    except Exception as exc:
        logger.exception("Query enrichment error")
        raise HTTPException(status_code=500, detail=str(exc))

