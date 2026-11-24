"""Database session factory with automatic SQLite fallback."""

from __future__ import annotations

import logging
from typing import Tuple
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings

logger = logging.getLogger("freshweb_rag")


def create_engine_and_session(db_url: str):
    """Create an async SQLAlchemy engine and sessionmaker suitable for the given database URL."""
    connect_args = {}
    engine_kwargs = {"echo": False, "future": True}

    if "sqlite" in db_url:
        connect_args["check_same_thread"] = False
        engine_kwargs["connect_args"] = connect_args
        if ":memory:" in db_url:
            engine_kwargs["poolclass"] = StaticPool
    else:
        engine_kwargs["pool_pre_ping"] = True
        engine_kwargs["pool_size"] = 10
        engine_kwargs["max_overflow"] = 20

    eng = create_async_engine(db_url, **engine_kwargs)
    session_factory = sessionmaker(
        bind=eng,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )
    return eng, session_factory


# Initial engine configured from settings
engine, AsyncSessionLocal = create_engine_and_session(settings.database_url)
