"""Conversation history endpoints."""

from __future__ import annotations

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Conversation, ConversationMessage
from app.dependencies import get_db, get_current_user_id

logger = logging.getLogger("freshweb_rag")
router = APIRouter()


class ConversationCreate(BaseModel):
    title: Optional[str] = None


class ConversationInfo(BaseModel):
    id: UUID
    title: Optional[str]
    created_at: str
    updated_at: str
    is_active: bool


class MessageInfo(BaseModel):
    id: UUID
    role: str
    content: str
    created_at: str


class ConversationHistory(BaseModel):
    conversation: ConversationInfo
    messages: List[MessageInfo]


@router.post("/conversations", response_model=ConversationInfo, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: ConversationCreate,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    conv = Conversation(user_id=user_id, title=body.title)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)

    # MongoDB persistence
    try:
        from app.database.mongodb import mongodb_manager
        await mongodb_manager.create_conversation(user_id=user_id, title=body.title)
    except Exception as m_exc:
        logger.debug("MongoDB create_conversation sync skipped: %s", m_exc)
    return ConversationInfo(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at.isoformat(),
        updated_at=conv.updated_at.isoformat(),
        is_active=conv.is_active,
    )


@router.get("/conversations", response_model=List[ConversationInfo])
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id, Conversation.is_active == True)
        .order_by(Conversation.updated_at.desc())
    )
    convs = result.scalars().all()
    return [
        ConversationInfo(
            id=c.id,
            title=c.title,
            created_at=c.created_at.isoformat(),
            updated_at=c.updated_at.isoformat(),
            is_active=c.is_active,
        )
        for c in convs
    ]


@router.get("/conversations/{conversation_id}/history", response_model=ConversationHistory)
async def get_conversation_history(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    conv = await db.get(Conversation, conversation_id)
    if not conv or str(conv.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    result = await db.execute(
        select(ConversationMessage)
        .where(ConversationMessage.conversation_id == conversation_id)
        .order_by(ConversationMessage.created_at.asc())
    )
    msgs = result.scalars().all()
    return ConversationHistory(
        conversation=ConversationInfo(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at.isoformat(),
            updated_at=conv.updated_at.isoformat(),
            is_active=conv.is_active,
        ),
        messages=[
            MessageInfo(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at.isoformat(),
            )
            for m in msgs
        ],
    )

