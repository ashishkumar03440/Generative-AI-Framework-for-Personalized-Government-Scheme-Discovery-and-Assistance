"""SQLAlchemy ORM models for FreshWeb-RAG.

Supports both PostgreSQL (production) and SQLite (zero-config local development).
"""

from __future__ import annotations

import datetime
import uuid
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.types import TypeDecorator, CHAR, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB as PG_JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class GUID(TypeDecorator):
    """Platform-independent GUID/UUID type.
    Uses PostgreSQL's native UUID type when available, otherwise CHAR(36).
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            return uuid.UUID(str(value))
        return value


# Use JSONB on PostgreSQL for indexed performance, standard JSON on other dialects
JSON_TYPE = JSON().with_variant(PG_JSONB, "postgresql")


# ── Users ──────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(GUID, primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(128), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


# ── Documents ──────────────────────────────────────────────────────────────────
class Document(Base):
    __tablename__ = "documents"

    id = Column(GUID, primary_key=True, default=uuid4)
    user_id = Column(GUID, nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    content_type = Column(String(128), nullable=True)
    file_size = Column(Integer, nullable=True)
    chunk_count = Column(Integer, default=0)
    status = Column(String(32), default="pending")  # pending, indexed, failed
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    indexed_at = Column(DateTime, nullable=True)
    metadata_ = Column("metadata", JSON_TYPE, default=dict)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(GUID, primary_key=True, default=uuid4)
    document_id = Column(GUID, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(GUID, nullable=False, index=True)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    source = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_verified_at = Column(DateTime, nullable=True)
    metadata_ = Column("metadata", JSON_TYPE, default=dict)
    qdrant_point_id = Column(String(64), nullable=True)

    document = relationship("Document", back_populates="chunks")


# ── Conversations ──────────────────────────────────────────────────────────────
class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(GUID, primary_key=True, default=uuid4)
    user_id = Column(GUID, nullable=False, index=True)
    title = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)
    metadata_ = Column("metadata", JSON_TYPE, default=dict)

    messages = relationship("ConversationMessage", back_populates="conversation", cascade="all, delete-orphan")


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(GUID, primary_key=True, default=uuid4)
    conversation_id = Column(GUID, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(GUID, nullable=False)
    role = Column(String(32), nullable=False)  # user | assistant | system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    metadata_ = Column("metadata", JSON_TYPE, default=dict)

    conversation = relationship("Conversation", back_populates="messages")


# ── Memory ─────────────────────────────────────────────────────────────────────
class MemoryRecord(Base):
    __tablename__ = "memory_records"

    id = Column(GUID, primary_key=True, default=uuid4)
    user_id = Column(GUID, nullable=False, index=True)
    memory_type = Column(String(64), nullable=False, index=True)  # episodic|semantic|procedural|short_term
    content = Column(Text, nullable=False)
    importance_score = Column(Float, default=0.5)
    confidence_score = Column(Float, default=0.8)
    source_conversation_id = Column(GUID, nullable=True)
    provenance = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)
    metadata_ = Column("metadata", JSON_TYPE, default=dict)
    qdrant_point_id = Column(String(64), nullable=True)


# ── Analytics ──────────────────────────────────────────────────────────────────
class QueryAnalyticsRecord(Base):
    __tablename__ = "query_analytics"

    id = Column(GUID, primary_key=True, default=uuid4)
    user_id = Column(GUID, nullable=True)
    conversation_id = Column(GUID, nullable=True)
    query_hash = Column(String(64), nullable=True)
    complexity = Column(String(32), nullable=True)
    freshness_required = Column(Boolean, nullable=True)
    memory_intent = Column(String(64), nullable=True)
    rag_route = Column(String(64), nullable=True)
    retrieval_verdict = Column(String(32), nullable=True)
    num_chunks_retrieved = Column(Integer, default=0)
    num_web_sources = Column(Integer, default=0)
    freshness_score = Column(Float, nullable=True)
    reliability_score = Column(Float, nullable=True)
    verification_result = Column(String(32), nullable=True)
    llm_input_tokens = Column(Integer, default=0)
    llm_output_tokens = Column(Integer, default=0)
    estimated_cost_usd = Column(Float, default=0.0)
    cache_hit = Column(Boolean, default=False)
    execution_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
