"""Health check route."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
import datetime

router = APIRouter()


@router.get("/health", response_class=JSONResponse, summary="Service health check")
async def health_check():
    return {
        "status": "ok",
        "service": "FreshWeb-RAG",
        "version": "1.0.0",
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


@router.get("/config", summary="Public runtime configuration")
async def get_config():
    from app.config import settings
    return {
        "app_env": settings.app_env,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "memory_enabled": settings.memory_enabled,
        "top_k": settings.top_k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
