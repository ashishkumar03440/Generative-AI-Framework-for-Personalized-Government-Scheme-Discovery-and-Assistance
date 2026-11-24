"""Hybrid retriever — dense + sparse (BM25) with Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
from typing import List, Optional

from app.config import settings

logger = logging.getLogger("freshweb_rag")


async def hybrid_search(
    query: str,
    user_id: str,
    top_k: int = None,
    search_type: str = "hybrid",
) -> List[dict]:
    """
    Run hybrid retrieval combining dense vector search and BM25 sparse search.

    Returns a list of result dicts with keys: chunk_id, document_id, content, score, source, page_number.
    """
    _top_k = top_k or settings.top_k

    if search_type == "dense":
        return await _dense_search(query, user_id, _top_k)
    elif search_type == "sparse":
        return await _sparse_search(query, user_id, _top_k)
    else:
        return await _hybrid_rrf(query, user_id, _top_k)


async def _dense_search(query: str, user_id: str, top_k: int) -> List[dict]:
    """Dense vector similarity search."""
    from app.utils.llm_client import embed_query
    from app.retrieval.static.vector_store import search_vectors

    query_vec = await embed_query(query)
    raw = await search_vectors(
        collection=settings.qdrant_collection_documents,
        query_vector=query_vec,
        top_k=top_k,
        filter_conditions={"user_id": user_id},
    )
    return [_format_result(r) for r in raw]


async def _sparse_search(query: str, user_id: str, top_k: int) -> List[dict]:
    """
    BM25 sparse retrieval over stored chunks.
    For now, retrieves candidates from Qdrant and re-ranks with BM25.
    """
    from app.utils.llm_client import embed_query
    from app.retrieval.static.vector_store import search_vectors

    # Fetch more candidates for BM25 reranking
    query_vec = await embed_query(query)
    candidates = await search_vectors(
        collection=settings.qdrant_collection_documents,
        query_vector=query_vec,
        top_k=min(top_k * 4, 50),
        filter_conditions={"user_id": user_id},
    )

    if not candidates:
        return []

    try:
        from rank_bm25 import BM25Okapi
        import re

        def tokenize(text: str) -> List[str]:
            return re.findall(r"\w+", text.lower())

        corpus = [c["payload"].get("content", "") for c in candidates]
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenize(query))

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [_format_result(r, score=s) for r, s in ranked]

    except ImportError:
        logger.warning("rank-bm25 not installed, returning dense results")
        return [_format_result(c) for c in candidates[:top_k]]


async def _hybrid_rrf(query: str, user_id: str, top_k: int) -> List[dict]:
    """Reciprocal Rank Fusion of dense and sparse results."""
    dense = await _dense_search(query, user_id, top_k)
    sparse = await _sparse_search(query, user_id, top_k)

    # RRF fusion
    k = 60  # RRF constant
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for rank, item in enumerate(dense):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        items[cid] = item

    for rank, item in enumerate(sparse):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        items[cid] = item

    ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
    results = []
    for cid in ranked_ids:
        item = items[cid]
        item["score"] = round(scores[cid], 6)
        results.append(item)

    return results


def _format_result(raw: dict, score: float = None) -> dict:
    payload = raw.get("payload", {})
    return {
        "chunk_id": raw.get("id", ""),
        "document_id": payload.get("document_id", ""),
        "content": payload.get("content", ""),
        "score": score if score is not None else raw.get("score", 0.0),
        "source": payload.get("source", ""),
        "page_number": payload.get("page_number"),
    }

