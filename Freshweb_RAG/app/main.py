"""Main entry point for FreshWeb-RAG FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings

logger = logging.getLogger("freshweb_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown lifecycle."""
    logger.info("Starting FreshWeb-RAG backend (env=%s)", settings.app_env)

    # ── MongoDB ───────────────────────────────────────────────────────────────
    try:
        from app.database.mongodb import mongodb_manager
        await mongodb_manager.connect()
    except Exception as exc:
        logger.warning("MongoDB init skipped: %s", exc)

    # ── Database tables (SQL fallback) ────────────────────────────────────────
    try:
        from app.database import session as db_session
        from app.database.models import Base
        async with db_session.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables ready (%s)", db_session.engine.url)
    except Exception as exc:
        if settings.app_env == "development" and "postgresql" in settings.database_url:
            from pathlib import Path
            logger.warning("PostgreSQL offline at %s. Switching to zero-config SQLite: data/freshweb.db", settings.database_url)
            fallback_path = Path("data/freshweb.db").resolve()
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            fallback_url = f"sqlite+aiosqlite:///{fallback_path}"
            db_session.engine, db_session.AsyncSessionLocal = db_session.create_engine_and_session(fallback_url)
            async with db_session.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Local SQLite database tables ready at %s", fallback_path)
        else:
            logger.warning("Database init skipped: %s", exc)

    # ── Qdrant collections ────────────────────────────────────────────────────
    try:
        from app.dependencies import ensure_qdrant_collections
        await ensure_qdrant_collections()
        logger.info("Qdrant collections ready")
    except Exception as exc:
        logger.warning("Qdrant init skipped: %s", exc)

    yield

    logger.info("Shutting down FreshWeb-RAG backend")
    try:
        from app.database.mongodb import mongodb_manager
        await mongodb_manager.close()
    except Exception:
        pass


app = FastAPI(
    title="FreshWeb-RAG Backend",
    version="1.0.0",
    description=(
        "Adaptive, Memory-Augmented, Web-Aware Retrieval Augmented Generation backend. "
        "Combines live web intelligence, long-term memory, and corrective retrieval."
    ),
    debug=settings.app_env == "development",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate limiting (slowapi) ───────────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("Rate limiting enabled (%d req/min)", settings.rate_limit_per_minute)
except ImportError:
    logger.warning("slowapi not installed; rate limiting disabled.")

# ── Routers ───────────────────────────────────────────────────────────────────
from app.api.routes import (  # noqa: E402
    auth,
    health,
    chat,
    documents,
    memory,
    conversations,
    search,
    analytics,
    memory_aware,
)

API_PREFIX = "/api/v1"
app.include_router(auth.router,          prefix=API_PREFIX, tags=["Authentication"])
app.include_router(health.router,        prefix=API_PREFIX, tags=["Health"])
app.include_router(chat.router,          prefix=API_PREFIX, tags=["Chat"])
app.include_router(documents.router,     prefix=API_PREFIX, tags=["Documents"])
app.include_router(memory.router,        prefix=API_PREFIX, tags=["Memory"])
app.include_router(conversations.router, prefix=API_PREFIX, tags=["Conversations"])
app.include_router(search.router,        prefix=API_PREFIX, tags=["Search"])
app.include_router(analytics.router,     prefix=API_PREFIX, tags=["Analytics"])
app.include_router(memory_aware.router,  prefix=API_PREFIX, tags=["Memory-Aware Routing"])


# ── Static Web UI & Documentation ──────────────────────────────────────────────
from pathlib import Path
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def serve_control_center():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>FreshWeb-RAG Backend Online</h1><p><a href='/docs'>Swagger API Docs</a></p>")


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
    )
