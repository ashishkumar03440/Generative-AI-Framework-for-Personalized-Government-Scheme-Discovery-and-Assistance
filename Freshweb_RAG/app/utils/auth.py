"""JWT authentication utilities for FreshWeb-RAG.

Supports both development (open access) and production (JWT bearer token) modes.
Never trust user_id from request body — always derive from verified token.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

logger = logging.getLogger("freshweb_rag")
security = HTTPBearer(auto_error=False)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT access token."""
    try:
        from jose import jwt
    except ImportError:
        raise RuntimeError("python-jose not installed. Run: pip install python-jose[cryptography]")

    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {"sub": user_id, "exp": expire, "iat": datetime.utcnow()}
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[str]:
    """Decode and verify a JWT token. Returns user_id or None."""
    try:
        from jose import jwt, JWTError
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        return user_id
    except Exception as exc:
        logger.debug("Token decode failed: %s", exc)
        return None


# ── Dev demo tokens ───────────────────────────────────────────────────────────
_DEMO_TOKEN = "demo-token"
_DEMO_USER_ID = "00000000-0000-0000-0000-000000000001"


async def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    """
    FastAPI dependency: extract and verify user_id from Bearer token.

    - In development mode: accepts 'demo-token' and unauthenticated requests.
    - In production mode: requires a valid JWT Bearer token.
    """
    if credentials:
        token = credentials.credentials

        # Allow demo token in development
        if settings.app_env == "development" and token == _DEMO_TOKEN:
            return _DEMO_USER_ID

        # Try JWT decode
        user_id = decode_access_token(token)
        if user_id:
            return user_id

        # Token present but invalid
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # No credentials at all
    if settings.app_env == "development":
        return _DEMO_USER_ID  # Allow unauthenticated in dev

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required.",
        headers={"WWW-Authenticate": "Bearer"},
    )

