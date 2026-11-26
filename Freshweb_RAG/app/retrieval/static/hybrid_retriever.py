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
    Includes database chunk fallback if vector store is unavailable or empty.

    Returns a list of result dicts with keys: chunk_id, document_id, content, score, source, page_number.
    """
    _top_k = top_k or settings.top_k
    results: List[dict] = []

    try:
        if search_type == "dense":
            results = await _dense_search(query, user_id, _top_k)
        elif search_type == "sparse":
            results = await _sparse_search(query, user_id, _top_k)
        else:
            results = await _hybrid_rrf(query, user_id, _top_k)
    except Exception as exc:
        logger.warning("Vector search failed or offline (%s). Using database fallback.", exc)
        results = []

    # If vector search returned no results, fall back to direct database chunk search
    if not results:
        results = await _database_fallback_search(query, user_id, _top_k)

    return results


async def _database_fallback_search(query: str, user_id: str, top_k: int) -> List[dict]:
    """Search chunks directly from database (MongoDB and SQLite) using keyword and meta matching."""
    import re

    candidates: List[dict] = []

    # 1. Try MongoDB
    try:
        from app.database.mongodb import mongodb_manager
        if mongodb_manager.is_connected and mongodb_manager.db is not None:
            m_chunks = await mongodb_manager.db.document_chunks.find({"user_id": user_id}).to_list(length=100)
            for c in m_chunks:
                candidates.append({
                    "chunk_id": str(c.get("id") or c.get("_id", "")),
                    "document_id": str(c.get("document_id", "")),
                    "content": c.get("content", ""),
                    "score": 0.5,
                    "source": c.get("source", "MongoDB"),
                    "page_number": c.get("page_number", 1),
                })
        else:
            # Check MongoDB in-memory fallback
            for c in mongodb_manager._fallback.document_chunks:
                if c.get("user_id") == user_id:
                    candidates.append({
                        "chunk_id": str(c.get("id", "")),
                        "document_id": str(c.get("document_id", "")),
                        "content": c.get("content", ""),
                        "score": 0.5,
                        "source": c.get("source", "Document"),
                        "page_number": c.get("page_number", 1),
                    })
    except Exception as m_err:
        logger.debug("MongoDB chunk fallback lookup skipped: %s", m_err)

    # 2. Try SQLite databases if no chunks found yet
    if not candidates:
        import os, sqlite3
        db_paths = ["data/freshweb.db", "Freshweb_RAG/data/freshweb.db", "../data/freshweb.db"]
        for path in db_paths:
            if os.path.exists(path):
                try:
                    conn = sqlite3.connect(path)
                    cur = conn.cursor()
                    rows = cur.execute(
                        "SELECT id, document_id, content, page_number, source FROM document_chunks WHERE user_id = ? LIMIT 100",
                        (user_id,),
                    ).fetchall()
                    conn.close()
                    for r in rows:
                        candidates.append({
                            "chunk_id": str(r[0]),
                            "document_id": str(r[1]),
                            "content": r[2],
                            "score": 0.5,
                            "source": r[4] or "Document",
                            "page_number": r[3] or 1,
                        })
                    if candidates:
                        break
                except Exception as sq_err:
                    logger.debug("SQLite chunk fallback error for %s: %s", path, sq_err)

    if not candidates:
        return []

    # Score candidates using token overlap or return top if meta query
    q_tokens = set(re.findall(r"\w+", query.lower()))
    stop_words = {"what", "is", "there", "any", "in", "my", "the", "do", "i", "have", "are", "a", "an", "of", "to"}
    keywords = q_tokens - stop_words

    # If it's a general meta-query (e.g. "what is in my documents", "is there any document"), return the initial chunks
    if not keywords or any(word in query.lower() for word in ["any document", "my document", "knowledge base", "knowledgebase", "uploaded", "my file", "all"]):
        for i, c in enumerate(candidates[:top_k]):
            c["score"] = 0.85 - (i * 0.05)
        return candidates[:top_k]

    # Keyword matching
    scored = []
    for c in candidates:
        text = c["content"].lower()
        match_count = sum(1 for kw in keywords if kw in text)
        score = match_count / len(keywords) if keywords else 0.5
        c_copy = dict(c)
        c_copy["score"] = score
        scored.append(c_copy)

    scored.sort(key=lambda x: x["score"], reverse=True)
    best = [s for s in scored if s["score"] > 0]
    return best[:top_k] if best else candidates[:top_k]


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

