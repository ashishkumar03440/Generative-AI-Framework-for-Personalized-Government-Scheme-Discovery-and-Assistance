"""Analytics endpoints."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import QueryAnalyticsRecord
from app.dependencies import get_db, get_current_user_id

logger = logging.getLogger("freshweb_rag")
router = APIRouter()


@router.get("/analytics", summary="Query analytics summary")
async def get_analytics(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    result = await db.execute(
        select(QueryAnalyticsRecord)
        .where(QueryAnalyticsRecord.user_id == user_id)
        .order_by(QueryAnalyticsRecord.created_at.desc())
        .limit(limit)
    )
    records = result.scalars().all()

    # Aggregate summary
    routes: dict[str, int] = {}
    total_cost = 0.0
    total_time_ms = 0.0
    cache_hits = 0

    for r in records:
        if r.rag_route:
            routes[r.rag_route] = routes.get(r.rag_route, 0) + 1
        total_cost += r.estimated_cost_usd or 0.0
        total_time_ms += r.execution_time_ms or 0.0
        if r.cache_hit:
            cache_hits += 1

    n = len(records)
    return {
        "total_queries": n,
        "route_distribution": routes,
        "total_estimated_cost_usd": round(total_cost, 6),
        "avg_execution_time_ms": round(total_time_ms / n, 2) if n > 0 else 0,
        "cache_hit_rate": round(cache_hits / n, 3) if n > 0 else 0,
        "recent_records": [
            {
                "id": str(r.id),
                "complexity": r.complexity,
                "route": r.rag_route,
                "verification": r.verification_result,
                "execution_time_ms": r.execution_time_ms,
                "created_at": r.created_at.isoformat(),
            }
            for r in records[:20]
        ],
    }

