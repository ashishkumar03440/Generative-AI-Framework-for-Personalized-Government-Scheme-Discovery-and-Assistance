"""Redis cache utility with in-memory local fallback."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger("freshweb_rag")

# In-memory dictionary fallback: key -> (expire_at, value)
_in_memory_cache: dict[str, tuple[float, Any]] = {}


def _make_cache_key(*parts: str) -> str:
    raw = ":".join(str(p) for p in parts if p)
    return "freshweb:" + hashlib.sha256(raw.encode()).hexdigest()[:32]


async def cache_get(key: str) -> Optional[Any]:
    # 1. Try Redis
    try:
        from app.dependencies import get_redis
        redis = get_redis()
        if redis:
            val = await redis.get(key)
            if val:
                return json.loads(val)
    except Exception:
        pass

    # 2. In-memory fallback
    if key in _in_memory_cache:
        expire_at, val = _in_memory_cache[key]
        if time.time() < expire_at:
            return val
        else:
            del _in_memory_cache[key]

    return None


async def cache_set(key: str, value: Any, ttl: int = None) -> None:
    _ttl = ttl or settings.cache_ttl_default

    # 1. Try Redis
    try:
        from app.dependencies import get_redis
        redis = get_redis()
        if redis:
            await redis.setex(key, _ttl, json.dumps(value, default=str))
            return
    except Exception:
        pass

    # 2. In-memory fallback
    _in_memory_cache[key] = (time.time() + _ttl, value)


async def cache_delete(key: str) -> None:
    try:
        from app.dependencies import get_redis
        redis = get_redis()
        if redis:
            await redis.delete(key)
    except Exception:
        pass

    _in_memory_cache.pop(key, None)


def query_cache_key(user_id: str, query: str, provider: str = "") -> str:
    return _make_cache_key("query", user_id, query, provider, settings.llm_model)


def memory_cache_key(user_id: str, query: str) -> str:
    return _make_cache_key("memory", user_id, query)


def web_search_cache_key(query: str) -> str:
    return _make_cache_key("websearch", query)
