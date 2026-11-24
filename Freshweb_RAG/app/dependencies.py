"""Dependency injection — DB, Redis, Qdrant, Auth.

Includes zero-friction local development fallbacks when PostgreSQL, Redis,
or Qdrant external containers are not running on localhost.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

logger = logging.getLogger("freshweb_rag")
security = HTTPBearer(auto_error=False)

# ── Database ──────────────────────────────────────────────────────────────────
from app.database import session as db_session


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a SQLAlchemy async session using current active engine."""
    async with db_session.AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


# ── Redis ─────────────────────────────────────────────────────────────────────
import redis.asyncio as aioredis  # noqa: E402

_redis_client: Optional[aioredis.Redis] = None


def get_redis() -> Optional[aioredis.Redis]:
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=1.5,
            )
        except Exception as exc:
            logger.debug("Redis client init failed: %s", exc)
            return None
    return _redis_client


# ── Qdrant ────────────────────────────────────────────────────────────────────
from qdrant_client import AsyncQdrantClient  # noqa: E402
from qdrant_client.models import Distance, VectorParams  # noqa: E402

_qdrant_client: Optional[AsyncQdrantClient] = None
_qdrant_is_local: bool = False


def get_qdrant() -> AsyncQdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        # Start with configured URL
        _qdrant_client = AsyncQdrantClient(url=settings.qdrant_url, timeout=3.0)
    return _qdrant_client


def set_local_qdrant() -> AsyncQdrantClient:
    """Switch to in-memory/embedded Qdrant when remote Qdrant is offline."""
    global _qdrant_client, _qdrant_is_local
    logger.info("Initializing in-process Qdrant vector engine...")
    _qdrant_client = AsyncQdrantClient(location=":memory:")
    _qdrant_is_local = True
    return _qdrant_client


async def ensure_qdrant_collections() -> None:
    """Create Qdrant collections if they don't exist, falling back to in-memory engine if server is offline."""
    global _qdrant_client
    client = get_qdrant()
    collections = [
        settings.qdrant_collection_documents,
        settings.qdrant_collection_memory,
    ]

    try:
        # Test connection with short timeout
        for col_name in collections:
            exists = await client.collection_exists(col_name)
            if not exists:
                await client.create_collection(
                    collection_name=col_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )
        logger.info("Connected to remote Qdrant server at %s", settings.qdrant_url)
    except Exception as exc:
        if settings.app_env == "development":
            logger.warning("Remote Qdrant unreachable at %s (%s). Falling back to in-memory vector engine.", settings.qdrant_url, exc)
            client = set_local_qdrant()
            for col_name in collections:
                await client.create_collection(
                    collection_name=col_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )
            logger.info("In-memory Qdrant collections initialized: %s", collections)
        else:
            logger.warning("Could not initialize Qdrant collections: %s", exc)


# ── Auth ──────────────────────────────────────────────────────────────────────
DEMO_TOKEN_USER_MAP: dict[str, str] = {
    "demo-token": "00000000-0000-0000-0000-000000000001",
}


async def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    """Extract user_id from Bearer token. Falls back to demo user in development."""
    if credentials:
        if credentials.credentials in DEMO_TOKEN_USER_MAP:
            return DEMO_TOKEN_USER_MAP[credentials.credentials]
        try:
            from app.utils.auth import decode_access_token
            user_id = decode_access_token(credentials.credentials)
            if user_id:
                return user_id
        except Exception:
            pass
        if settings.app_env != "development":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication token",
            )

    if settings.app_env == "development":
        return "00000000-0000-0000-0000-000000000001"

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication token",
    )
