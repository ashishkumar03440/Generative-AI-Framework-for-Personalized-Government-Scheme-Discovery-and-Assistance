"""Search endpoint — direct KB search without full RAG pipeline."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user_id

logger = logging.getLogger("freshweb_rag")
router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048)
    top_k: int = Field(default=5, ge=1, le=20)
    search_type: str = Field(default="hybrid", description="dense | sparse | hybrid")
    metadata_filter: Optional[dict] = None


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    source: Optional[str] = None
    page_number: Optional[int] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    query: str
    search_type: str


@router.post("/search", response_model=SearchResponse, summary="Direct knowledge base search")
async def search(
    body: SearchRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    try:
        from app.retrieval.static.hybrid_retriever import hybrid_search
        results = await hybrid_search(
            query=body.query,
            user_id=user_id,
            top_k=body.top_k,
            search_type=body.search_type,
        )
        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            total=len(results),
            query=body.query,
            search_type=body.search_type,
        )
    except Exception as exc:
        logger.exception("Search error for user=%s", user_id)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

