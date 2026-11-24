"""Answer synthesis with strict grounding and citation generation."""

from __future__ import annotations

import logging
import re
import uuid
from typing import List, Optional

logger = logging.getLogger("freshweb_rag")


async def synthesize_answer(
    query: str,
    evidence_items: List[dict],
    memory_context: List[dict],
    route: str,
    conflict_report: Optional[dict] = None,
) -> dict:
    """
    Generate a grounded answer from evidence items.

    Returns answer text, citations, sources_used, limitations, confidence.
    """
    if not evidence_items and not memory_context:
        return _empty_response(query)

    try:
        return await _llm_synthesize(query, evidence_items, memory_context, route, conflict_report)
    except Exception as exc:
        logger.exception("Answer synthesis failed: %s", exc)
        return _fallback_response(query, evidence_items, exc)


async def _llm_synthesize(
    query: str,
    evidence_items: List[dict],
    memory_context: List[dict],
    route: str,
    conflict_report: Optional[dict],
) -> dict:
    from app.utils.llm_client import get_llm_client
    import json

    llm = get_llm_client()

    # Build evidence block with citation IDs
    evidence_block = ""
    citation_map = {}
    for i, item in enumerate(evidence_items[:8]):
        cid = f"[{i+1}]"
        source = item.get("url") or item.get("source") or "Internal Document"
        content = item.get("content", "")[:800]
        evidence_block += f"\n{cid} Source: {source}\nContent: {content}\n"
        citation_map[cid] = {
            "citation_id": cid,
            "source_type": item.get("source_type", "web"),
            "title": item.get("title", source),
            "url": item.get("url"),
            "snippet": content[:200],
            "published_at": item.get("published_at"),
            "reliability_score": item.get("reliability_score"),
            "freshness_score": item.get("freshness_score"),
        }

    # Memory context block
    memory_block = ""
    if memory_context:
        memory_block = "\n\nUser Memory Context (not verified external evidence):\n"
        for m in memory_context[:3]:
            memory_block += f"- {m.get('content', '')[:300]}\n"

    conflict_note = ""
    if conflict_report and conflict_report.get("conflicts"):
        conflict_note = f"\nNote: Conflicting evidence detected. {conflict_report.get('notes', '')}"

    prompt = f"""You are a knowledgeable, careful AI assistant. Answer the user's question using ONLY the provided evidence.

RULES:
1. Ground every factual claim in the provided evidence. Use inline citations like [1], [2].
2. Use memory context only for understanding intent and preferences — not as factual evidence.
3. If evidence is insufficient, explicitly say what is unknown.
4. Never fabricate URLs or citations.
5. Never invent facts not found in the evidence.
6. If sources conflict, acknowledge both perspectives and their sources.
7. Be concise but complete.

User Question: {query}

Evidence:
{evidence_block}
{memory_block}
{conflict_note}

Write a clear, grounded answer with inline citations. End with a brief statement of limitations if any."""

    response = await llm.ainvoke(prompt)
    answer_text = response.content if hasattr(response, "content") else str(response)

    # Extract which citations were used
    used_cids = set(re.findall(r"\[\d+\]", answer_text))
    citations = [v for k, v in citation_map.items() if k in used_cids]
    sources_used = list({c["url"] or c["title"] for c in citations})

    # Estimate confidence from evidence quality
    if evidence_items:
        avg_rel = sum(e.get("reliability_score", 0.5) for e in evidence_items[:5]) / min(len(evidence_items), 5)
        confidence = round(0.4 + avg_rel * 0.5, 3)
    else:
        confidence = 0.3

    limitations = []
    if not evidence_items:
        limitations.append("No external evidence retrieved.")
    if conflict_report and conflict_report.get("conflicts"):
        limitations.append("Conflicting evidence detected. See answer for details.")

    return {
        "answer": answer_text,
        "route": route,
        "confidence": confidence,
        "citations": citations,
        "sources_used": sources_used,
        "memories_used": memory_context,
        "limitations": limitations,
        "verification_status": "pending",
        "retrieval_metadata": {
            "num_evidence_items": len(evidence_items),
            "num_memory_items": len(memory_context),
        },
    }


def _empty_response(query: str) -> dict:
    return {
        "answer": "I don't have sufficient information to answer this question. Please try uploading relevant documents or rephrasing your question.",
        "route": "no_evidence",
        "confidence": 0.0,
        "citations": [],
        "sources_used": [],
        "memories_used": [],
        "limitations": ["No evidence available."],
        "verification_status": "unverified",
        "retrieval_metadata": {},
    }


def _fallback_response(query: str, evidence_items: List[dict], error: Exception) -> dict:
    return {
        "answer": "I encountered an error generating a complete answer. Based on retrieved documents, please review them directly.",
        "route": "error_fallback",
        "confidence": 0.1,
        "citations": [],
        "sources_used": [e.get("url") or e.get("source", "") for e in evidence_items[:3]],
        "memories_used": [],
        "limitations": [f"Answer generation error: {str(error)[:200]}"],
        "verification_status": "failed",
        "retrieval_metadata": {},
    }

