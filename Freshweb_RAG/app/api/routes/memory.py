"""Memory management endpoints."""

from __future__ import annotations

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.models import MemoryRecord
from app.dependencies import get_db, get_current_user_id

logger = logging.getLogger("freshweb_rag")
router = APIRouter()

VALID_MEMORY_TYPES = {"episodic", "semantic", "procedural", "short_term"}


class MemoryInfo(BaseModel):
    id: UUID
    memory_type: str
    content: str
    importance_score: float
    confidence_score: float
    created_at: str
    updated_at: str
    source_conversation_id: Optional[UUID] = None


class MemoryListResponse(BaseModel):
    memories: List[MemoryInfo]
    total: int


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    memory_types: Optional[List[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)


class MemoryUpdateRequest(BaseModel):
    content: Optional[str] = None
    importance_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class MemoryForgetRequest(BaseModel):
    memory_ids: List[UUID]


@router.get("/memory", response_model=MemoryListResponse, summary="List memories")
async def list_memories(
    memory_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    stmt = select(MemoryRecord).where(MemoryRecord.user_id == user_id, MemoryRecord.is_active == True)
    if memory_type:
        stmt = stmt.where(MemoryRecord.memory_type == memory_type)
    stmt = stmt.order_by(MemoryRecord.importance_score.desc())
    result = await db.execute(stmt)
    records = result.scalars().all()
    return MemoryListResponse(
        memories=[
            MemoryInfo(
                id=r.id,
                memory_type=r.memory_type,
                content=r.content,
                importance_score=r.importance_score or 0.5,
                confidence_score=r.confidence_score or 0.8,
                created_at=r.created_at.isoformat(),
                updated_at=r.updated_at.isoformat(),
                source_conversation_id=r.source_conversation_id,
            )
            for r in records
        ],
        total=len(records),
    )


@router.get("/memory/{memory_id}", response_model=MemoryInfo, summary="Get a memory")
async def get_memory(
    memory_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    rec = await db.get(MemoryRecord, memory_id)
    if not rec or str(rec.user_id) != user_id or not rec.is_active:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return MemoryInfo(
        id=rec.id,
        memory_type=rec.memory_type,
        content=rec.content,
        importance_score=rec.importance_score or 0.5,
        confidence_score=rec.confidence_score or 0.8,
        created_at=rec.created_at.isoformat(),
        updated_at=rec.updated_at.isoformat(),
        source_conversation_id=rec.source_conversation_id,
    )


@router.patch("/memory/{memory_id}", response_model=MemoryInfo, summary="Update a memory")
async def update_memory(
    memory_id: UUID,
    body: MemoryUpdateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    rec = await db.get(MemoryRecord, memory_id)
    if not rec or str(rec.user_id) != user_id or not rec.is_active:
        raise HTTPException(status_code=404, detail="Memory not found.")
    if body.content is not None:
        rec.content = body.content
    if body.importance_score is not None:
        rec.importance_score = body.importance_score
    await db.commit()
    await db.refresh(rec)
    return MemoryInfo(
        id=rec.id,
        memory_type=rec.memory_type,
        content=rec.content,
        importance_score=rec.importance_score or 0.5,
        confidence_score=rec.confidence_score or 0.8,
        created_at=rec.created_at.isoformat(),
        updated_at=rec.updated_at.isoformat(),
        source_conversation_id=rec.source_conversation_id,
    )


@router.delete("/memory/{memory_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a memory")
async def delete_memory(
    memory_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    rec = await db.get(MemoryRecord, memory_id)
    if not rec or str(rec.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Memory not found.")
    rec.is_active = False
    await db.commit()
    # Remove Qdrant vector
    try:
        from app.memory.memory_store import delete_memory_vector
        if rec.qdrant_point_id:
            await delete_memory_vector(rec.qdrant_point_id)
    except Exception as exc:
        logger.warning("Could not remove memory vector: %s", exc)


@router.post("/memory/search", summary="Semantic memory search")
async def search_memory(
    body: MemorySearchRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    try:
        from app.memory.memory_retriever import retrieve_memories
        results = await retrieve_memories(
            query=body.query,
            user_id=user_id,
            memory_types=body.memory_types,
            top_k=body.top_k,
            db=db,
        )
        return {"results": results, "total": len(results)}
    except Exception as exc:
        logger.exception("Memory search error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/memory/forget", status_code=status.HTTP_200_OK, summary="Bulk delete memories")
async def forget_memories(
    body: MemoryForgetRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    deleted = 0
    for mid in body.memory_ids:
        rec = await db.get(MemoryRecord, mid)
        if rec and str(rec.user_id) == user_id:
            rec.is_active = False
            deleted += 1
    await db.commit()
    return {"deleted": deleted}

