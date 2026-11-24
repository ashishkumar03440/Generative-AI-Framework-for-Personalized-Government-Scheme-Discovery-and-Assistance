"""Initial schema migration for FreshWeb-RAG.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-11-12 16:10:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Documents ─────────────────────────────────────────────────────────────
    op.create_table(
        'documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('filename', sa.String(length=512), nullable=False),
        sa.Column('content_type', sa.String(length=128), nullable=True),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('chunk_count', sa.Integer(), server_default='0', nullable=True),
        sa.Column('status', sa.String(length=32), server_default='pending', nullable=True),
        sa.Column('uploaded_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('indexed_at', sa.DateTime(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), server_default='{}', nullable=True),
    )

    # ── Document Chunks ───────────────────────────────────────────────────────
    op.create_table(
        'document_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=True),
        sa.Column('source', sa.String(length=512), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('last_verified_at', sa.DateTime(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), server_default='{}', nullable=True),
        sa.Column('qdrant_point_id', sa.String(length=64), nullable=True),
    )

    # ── Conversations ─────────────────────────────────────────────────────────
    op.create_table(
        'conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('title', sa.String(length=512), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=True),
        sa.Column('metadata', postgresql.JSONB(), server_default='{}', nullable=True),
    )

    # ── Conversation Messages ─────────────────────────────────────────────────
    op.create_table(
        'conversation_messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role', sa.String(length=32), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), server_default='{}', nullable=True),
    )

    # ── Memory Records ────────────────────────────────────────────────────────
    op.create_table(
        'memory_records',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('memory_type', sa.String(length=64), nullable=False, index=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('importance_score', sa.Float(), server_default='0.5', nullable=True),
        sa.Column('confidence_score', sa.Float(), server_default='0.8', nullable=True),
        sa.Column('source_conversation_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('provenance', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=True),
        sa.Column('metadata', postgresql.JSONB(), server_default='{}', nullable=True),
        sa.Column('qdrant_point_id', sa.String(length=64), nullable=True),
    )

    # ── Query Analytics ───────────────────────────────────────────────────────
    op.create_table(
        'query_analytics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('query_hash', sa.String(length=64), nullable=True),
        sa.Column('complexity', sa.String(length=32), nullable=True),
        sa.Column('freshness_required', sa.Boolean(), nullable=True),
        sa.Column('memory_intent', sa.String(length=64), nullable=True),
        sa.Column('rag_route', sa.String(length=64), nullable=True),
        sa.Column('retrieval_verdict', sa.String(length=32), nullable=True),
        sa.Column('num_chunks_retrieved', sa.Integer(), server_default='0', nullable=True),
        sa.Column('num_web_sources', sa.Integer(), server_default='0', nullable=True),
        sa.Column('freshness_score', sa.Float(), nullable=True),
        sa.Column('reliability_score', sa.Float(), nullable=True),
        sa.Column('verification_result', sa.String(length=32), nullable=True),
        sa.Column('llm_input_tokens', sa.Integer(), server_default='0', nullable=True),
        sa.Column('llm_output_tokens', sa.Integer(), server_default='0', nullable=True),
        sa.Column('estimated_cost_usd', sa.Float(), server_default='0.0', nullable=True),
        sa.Column('cache_hit', sa.Boolean(), server_default='false', nullable=True),
        sa.Column('execution_time_ms', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('query_analytics')
    op.drop_table('memory_records')
    op.drop_table('conversation_messages')
    op.drop_table('conversations')
    op.drop_table('document_chunks')
    op.drop_table('documents')

