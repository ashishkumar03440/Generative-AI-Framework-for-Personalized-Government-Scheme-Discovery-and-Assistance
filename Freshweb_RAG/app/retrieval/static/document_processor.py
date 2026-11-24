"""Document processor — extracts, chunks, embeds, and indexes documents."""

from __future__ import annotations

import datetime
import logging
import uuid
from pathlib import Path
from typing import List, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

logger = logging.getLogger("freshweb_rag")


def _extract_text_pdf(content: bytes, filename: str) -> List[Tuple[str, int]]:
    """Extract text with page numbers from PDF."""
    try:
        import PyPDF2
        import io
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((text, i + 1))
        return pages
    except Exception as exc:
        logger.warning("PDF extraction failed: %s", exc)
        return [(content.decode("utf-8", errors="ignore"), 1)]


def _extract_text_docx(content: bytes, filename: str) -> List[Tuple[str, int]]:
    """Extract text from DOCX."""
    try:
        import docx
        import io
        doc = docx.Document(io.BytesIO(content))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [(text, 1)]
    except Exception as exc:
        logger.warning("DOCX extraction failed: %s", exc)
        return [(content.decode("utf-8", errors="ignore"), 1)]


def _extract_text_plain(content: bytes, filename: str) -> List[Tuple[str, int]]:
    return [(content.decode("utf-8", errors="ignore"), 1)]


def extract_text(content: bytes, filename: str) -> List[Tuple[str, int]]:
    """Route to appropriate extractor based on filename extension."""
    fn = filename.lower()
    if fn.endswith(".pdf"):
        return _extract_text_pdf(content, filename)
    elif fn.endswith(".docx"):
        return _extract_text_docx(content, filename)
    else:
        return _extract_text_plain(content, filename)


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[str]:
    """Split text into overlapping chunks."""
    _size = chunk_size or settings.chunk_size
    _overlap = chunk_overlap or settings.chunk_overlap

    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + _size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += _size - _overlap
        if start >= len(text):
            break
    return chunks


async def process_and_index_document(
    doc_id: str,
    user_id: str,
    content: bytes,
    filename: str,
    db: Optional[AsyncSession] = None,
) -> None:
    """Full pipeline: extract → chunk → embed → store in Qdrant + DB."""
    from app.database.models import Document, DocumentChunk
    from app.retrieval.static.vector_store import upsert_vectors
    from app.utils.llm_client import embed_texts
    from app.database import session as db_session

    logger.info("Processing document %s (%s)", doc_id, filename)
    local_session = None
    if db is None:
        local_session = db_session.AsyncSessionLocal()
        db = local_session

    try:
        # 1. Extract
        pages = extract_text(content, filename)
        all_chunks: List[dict] = []

        for page_text, page_num in pages:
            for chunk_text_item in chunk_text(page_text):
                all_chunks.append({
                    "content": chunk_text_item,
                    "page_number": page_num,
                })

        if not all_chunks:
            logger.warning("No chunks extracted from document %s", doc_id)
            return

        # 2. Embed
        texts = [c["content"] for c in all_chunks]
        vectors = await embed_texts(texts)

        # 3. Build Qdrant points
        point_ids = [str(uuid.uuid4()) for _ in all_chunks]
        payloads = [
            {
                "document_id": doc_id,
                "user_id": user_id,
                "content": all_chunks[i]["content"],
                "page_number": all_chunks[i]["page_number"],
                "source": filename,
                "chunk_index": i,
            }
            for i in range(len(all_chunks))
        ]

        await upsert_vectors(
            collection=settings.qdrant_collection_documents,
            ids=point_ids,
            vectors=vectors,
            payloads=payloads,
        )

        # 4. Store chunks in DB
        chunk_records = [
            DocumentChunk(
                id=uuid.UUID(point_ids[i]),
                document_id=doc_id,
                user_id=user_id,
                content=all_chunks[i]["content"],
                chunk_index=i,
                page_number=all_chunks[i]["page_number"],
                source=filename,
                qdrant_point_id=point_ids[i],
            )
            for i in range(len(all_chunks))
        ]
        db.add_all(chunk_records)

        # 5. Update document status
        doc = await db.get(Document, uuid.UUID(doc_id))
        if doc:
            doc.status = "indexed"
            doc.chunk_count = len(all_chunks)
            doc.indexed_at = datetime.datetime.utcnow()

        await db.commit()
        logger.info("Indexed %d chunks for document %s", len(all_chunks), doc_id)

        # 6. MongoDB persistence
        try:
            from app.database.mongodb import mongodb_manager
            await mongodb_manager.update_document_status(
                doc_id=doc_id,
                status="indexed",
                chunk_count=len(all_chunks),
                indexed_at=datetime.datetime.utcnow().isoformat(),
            )
            await mongodb_manager.insert_chunks([
                {
                    "id": point_ids[i],
                    "document_id": doc_id,
                    "user_id": user_id,
                    "content": all_chunks[i]["content"],
                    "chunk_index": i,
                    "page_number": all_chunks[i]["page_number"],
                    "source": filename,
                    "qdrant_point_id": point_ids[i],
                }
                for i in range(len(all_chunks))
            ])
        except Exception as m_exc:
            logger.debug("MongoDB chunk sync skipped: %s", m_exc)

    except Exception as exc:
        logger.exception("Failed to process document %s: %s", doc_id, exc)
        try:
            from app.database.models import Document
            doc = await db.get(Document, uuid.UUID(doc_id))
            if doc:
                doc.status = "failed"
            await db.commit()
        except Exception:
            pass
    finally:
        if local_session:
            await local_session.close()

