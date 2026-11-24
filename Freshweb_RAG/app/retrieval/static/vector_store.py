"""Vector store operations for Qdrant."""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from app.config import settings

logger = logging.getLogger("freshweb_rag")


async def upsert_vectors(
    collection: str,
    ids: List[str],
    vectors: List[List[float]],
    payloads: List[dict],
) -> None:
    """Upsert vectors into a Qdrant collection."""
    from qdrant_client.models import PointStruct
    from app.dependencies import get_qdrant

    client = get_qdrant()
    points = [
        PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
        for i in range(len(ids))
    ]
    await client.upsert(collection_name=collection, points=points)


async def search_vectors(
    collection: str,
    query_vector: List[float],
    top_k: int = 5,
    filter_conditions: Optional[dict] = None,
) -> List[dict]:
    """Search vectors by similarity."""
    from app.dependencies import get_qdrant
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = get_qdrant()
    qdrant_filter = None
    if filter_conditions:
        conditions = []
        for key, value in filter_conditions.items():
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        from qdrant_client.models import Filter
        qdrant_filter = Filter(must=conditions)

    results = await client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    )
    return [
        {
            "id": str(r.id),
            "score": r.score,
            "payload": r.payload,
        }
        for r in results
    ]


async def delete_document_vectors(document_id: str) -> None:
    """Delete all vectors belonging to a document."""
    from app.dependencies import get_qdrant
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = get_qdrant()
    await client.delete(
        collection_name=settings.qdrant_collection_documents,
        points_selector=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ),
    )
    logger.info("Deleted vectors for document_id=%s", document_id)

