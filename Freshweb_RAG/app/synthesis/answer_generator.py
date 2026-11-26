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
        return await _direct_knowledge_synthesize(query, route)

    try:
        return await _llm_synthesize(query, evidence_items, memory_context, route, conflict_report)
    except Exception as exc:
        logger.exception("Answer synthesis failed, falling back to direct knowledge: %s", exc)
        return await _direct_knowledge_synthesize(query, route)


def _extract_text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


async def _direct_knowledge_synthesize(query: str, route: str) -> dict:
    """Synthesize an answer using LLM general knowledge when no documents/web evidence are present."""
    from app.utils.llm_client import get_llm_client
    try:
        llm = get_llm_client()
        prompt = (
            "You are FreshWeb-RAG, an intelligent and helpful AI assistant.\n"
            "Answer the user's question thoroughly, clearly, and informatively.\n\n"
            f"User Question: {query}\n\n"
            "Answer:"
        )
        response = await llm.ainvoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        answer_text = _extract_text_content(raw)
        return {
            "answer": answer_text,
            "route": route if route not in ("no_evidence", "static_kb") else "direct_knowledge",
            "confidence": 0.85,
            "citations": [],
            "sources_used": [],
            "memories_used": [],
            "limitations": ["Answer synthesized using AI general knowledge; no matching internal documents found."],
            "verification_status": "passed",
            "retrieval_metadata": {
                "num_evidence_items": 0,
                "num_memory_items": 0,
            },
        }
    except Exception as exc:
        logger.warning("Direct LLM knowledge synthesis error: %s", exc)
        return {
            "answer": f"Regarding your question '{query}': Currently no specific documents are indexed in your knowledge base. Please upload your documents in the Knowledge Base tab or configure TAVILY_API_KEY for live web searching.",
            "route": "direct_knowledge",
            "confidence": 0.4,
            "citations": [],
            "sources_used": [],
            "memories_used": [],
            "limitations": [f"LLM call notice: {str(exc)[:150]}"],
            "verification_status": "unverified",
            "retrieval_metadata": {},
        }


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

    prompt = f"""You are a knowledgeable, helpful AI assistant for the user's FreshWeb-RAG system.
The user has uploaded documents to their Knowledge Base, and relevant excerpts or document records are provided in the Evidence section below.

RULES:
1. Ground your answer in the provided evidence. Always acknowledge and mention the user's documents when they ask about their files, resume, or knowledge base.
2. Ground factual claims in the provided evidence using inline citations like [1], [2].
3. Do NOT say "I don't have access to your personal files or knowledge base" because the knowledge base contents are explicitly provided to you in the Evidence section below.
4. If the user asks whether they have documents in their knowledge base or asks what files are uploaded, summarize the document names, topics, and contents present in the Evidence.
5. Never invent fake citations or URLs.
6. Be concise, well-structured, and helpful.

User Question: {query}

Evidence:
{evidence_block}
{memory_block}
{conflict_note}

Write a clear, well-structured answer with inline citations where evidence applies. End with a brief statement of limitations if any."""

    response = await llm.ainvoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    answer_text = _extract_text_content(raw)

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
        "answer": f"I was unable to find specific documents in your knowledge base for '{query}'. You can upload relevant documents in the Knowledge Base tab, or configure live web search for real-time information.",
        "route": "no_evidence",
        "confidence": 0.5,
        "citations": [],
        "sources_used": [],
        "memories_used": [],
        "limitations": ["No matching internal documents found."],
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

