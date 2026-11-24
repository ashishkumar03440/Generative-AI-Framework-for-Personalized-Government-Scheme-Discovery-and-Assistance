"""Document upload, listing, retrieval and deletion endpoints."""

from __future__ import annotations

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.models import Document
from app.dependencies import get_db, get_current_user_id

logger = logging.getLogger("freshweb_rag")
router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "application/octet-stream",
}
MAX_FILE_SIZE_MB = 50


class DocumentInfo(BaseModel):
    id: UUID
    filename: str
    content_type: Optional[str] = None
    file_size: Optional[int] = None
    chunk_count: int = 0
    status: str
    uploaded_at: str
    indexed_at: Optional[str] = None

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class DocumentUploadResponse(BaseModel):
    id: UUID
    filename: str
    status: str
    message: str


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document for indexing",
)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    # Validate content type
    ct = file.content_type or "application/octet-stream"
    # Accept by extension for clients that send wrong MIME
    fname = (file.filename or "").lower()
    if ct not in ALLOWED_CONTENT_TYPES and not any(
        fname.endswith(ext) for ext in [".pdf", ".docx", ".txt", ".md"]
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {ct}. Allowed: PDF, DOCX, TXT, Markdown.",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB.",
        )

    doc = Document(
        user_id=user_id,
        filename=file.filename,
        content_type=ct,
        file_size=len(content),
        status="pending",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # MongoDB persistence
    try:
        from app.database.mongodb import mongodb_manager
        await mongodb_manager.insert_document({
            "id": str(doc.id),
            "user_id": user_id,
            "filename": file.filename,
            "content_type": ct,
            "file_size": len(content),
            "status": "pending",
        })
    except Exception as m_exc:
        logger.debug("MongoDB insert_document sync skipped: %s", m_exc)

    # Trigger async indexing
    try:
        from app.retrieval.static.document_processor import process_and_index_document
        import asyncio
        asyncio.create_task(
            process_and_index_document(doc_id=str(doc.id), user_id=user_id, content=content, filename=file.filename, db=None)
        )
    except Exception as exc:
        logger.warning("Could not schedule indexing for doc %s: %s", doc.id, exc)

    return DocumentUploadResponse(
        id=doc.id,
        filename=doc.filename,
        status=doc.status,
        message="Document received and queued for indexing.",
    )


@router.get("/documents", response_model=DocumentListResponse, summary="List documents")
async def list_documents(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    result = await db.execute(
        select(Document).where(Document.user_id == user_id).order_by(Document.uploaded_at.desc())
    )
    docs = result.scalars().all()
    return DocumentListResponse(
        documents=[
            DocumentInfo(
                id=d.id,
                filename=d.filename,
                content_type=d.content_type,
                file_size=d.file_size,
                chunk_count=d.chunk_count or 0,
                status=d.status,
                uploaded_at=d.uploaded_at.isoformat(),
                indexed_at=d.indexed_at.isoformat() if d.indexed_at else None,
            )
            for d in docs
        ],
        total=len(docs),
    )


@router.get("/documents/{document_id}", response_model=DocumentInfo, summary="Get document info")
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    doc = await db.get(Document, document_id)
    if not doc or str(doc.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentInfo(
        id=doc.id,
        filename=doc.filename,
        content_type=doc.content_type,
        file_size=doc.file_size,
        chunk_count=doc.chunk_count or 0,
        status=doc.status,
        uploaded_at=doc.uploaded_at.isoformat(),
        indexed_at=doc.indexed_at.isoformat() if doc.indexed_at else None,
    )


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete document")
async def delete_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    doc = await db.get(Document, document_id)
    if not doc or str(doc.user_id) != user_id:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Remove from Qdrant
    try:
        from app.retrieval.static.vector_store import delete_document_vectors
        await delete_document_vectors(str(document_id))
    except Exception as exc:
        logger.warning("Could not remove vectors for doc %s: %s", document_id, exc)

    await db.delete(doc)
    await db.commit()

    # MongoDB delete
    try:
        from app.database.mongodb import mongodb_manager
        await mongodb_manager.delete_document(str(document_id), user_id)
    except Exception as m_exc:
        logger.debug("MongoDB delete_document sync skipped: %s", m_exc)
