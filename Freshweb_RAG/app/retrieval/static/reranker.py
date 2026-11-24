"""Cross-encoder reranker for improving retrieval quality."""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger("freshweb_rag")


async def rerank(
    query: str,
    chunks: List[dict],
    top_k: int = 3,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[dict]:
    """
    Rerank retrieved chunks using a cross-encoder model.

    Falls back to original order if sentence-transformers is unavailable.
    """
    if not chunks:
        return chunks

    try:
        return _cross_encoder_rerank(query, chunks, top_k, model_name)
    except ImportError:
        logger.warning("sentence-transformers not available for reranking. Using original order.")
        return chunks[:top_k]
    except Exception as exc:
        logger.warning("Cross-encoder reranking failed: %s. Using original order.", exc)
        return chunks[:top_k]


def _cross_encoder_rerank(
    query: str,
    chunks: List[dict],
    top_k: int,
    model_name: str,
) -> List[dict]:
    """Synchronous cross-encoder scoring."""
    from sentence_transformers import CrossEncoder

    model = _get_cross_encoder(model_name)
    pairs = [(query, chunk.get("content", "")[:512]) for chunk in chunks]
    scores = model.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    result = []
    for chunk, score in ranked[:top_k]:
        chunk = dict(chunk)
        chunk["rerank_score"] = float(score)
        result.append(chunk)
    return result


_ce_cache: dict = {}


def _get_cross_encoder(model_name: str):
    """Cache cross-encoder model instances."""
    if model_name not in _ce_cache:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder model: %s", model_name)
        _ce_cache[model_name] = CrossEncoder(model_name, max_length=512)
    return _ce_cache[model_name]

