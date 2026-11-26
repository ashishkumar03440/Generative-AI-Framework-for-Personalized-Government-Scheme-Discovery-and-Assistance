"""LangGraph node implementations."""

from __future__ import annotations

import logging
from typing import List

from app.graphs.graph_state import GraphState

logger = logging.getLogger("freshweb_rag")


# ── Node 1: Memory-Aware Analysis ────────────────────────────────────────────
async def memory_aware_analysis_node(state: GraphState) -> GraphState:
    """Classify query intent and detect historical references."""
    query = state["query"]
    user_id = state["user_id"]

    try:
        from app.memory_aware_router.router import analyze_memory_aware
        # We don't have db in graph state; use a fresh session
        analysis = await analyze_memory_aware(query=query, user_id=user_id, db=None)
        state["memory_intent"] = analysis.get("intent", "NEW_QUERY")
        state["has_historical_reference"] = analysis.get("has_historical_reference", False)
        state["reference_type"] = analysis.get("reference_type")
        state["requires_clarification"] = analysis.get("requires_clarification", False)
        state["enriched_query"] = query  # Will be updated if memory found
    except Exception as exc:
        logger.warning("Memory-aware analysis node error: %s", exc)
        state["memory_intent"] = "NEW_QUERY"
        state["has_historical_reference"] = False
        state["requires_clarification"] = False
        state["enriched_query"] = query

    return state


# ── Node 2: Memory Retrieval ──────────────────────────────────────────────────
async def memory_retrieval_node(state: GraphState) -> GraphState:
    """Retrieve relevant memories and enrich query."""
    query = state["query"]
    user_id = state["user_id"]

    try:
        from app.memory.memory_retriever import retrieve_memories
        memories = await retrieve_memories(query=query, user_id=user_id, top_k=5)
        state["memories_used"] = memories

        if memories:
            from app.memory_aware_router.query_rewriter import _llm_enrich
            context = "\n".join(f"- {m['content']}" for m in memories[:3])
            try:
                enriched = await _llm_enrich(query, context)
                state["enriched_query"] = enriched
            except Exception:
                state["enriched_query"] = query
        else:
            state["enriched_query"] = query
    except Exception as exc:
        logger.warning("Memory retrieval node error: %s", exc)
        state["memories_used"] = []
        state["enriched_query"] = query

    return state


# ── Node 3: Adaptive Query Analysis ──────────────────────────────────────────
async def query_analysis_node(state: GraphState) -> GraphState:
    """Classify complexity and freshness of the (potentially enriched) query."""
    query = state.get("enriched_query") or state["query"]

    try:
        from app.router.query_analyzer import analyze_query
        analysis = await analyze_query(query)
        state["complexity"] = analysis.complexity
        state["freshness_required"] = analysis.freshness_required
        state["sub_questions"] = analysis.sub_questions
        state["query_confidence"] = analysis.confidence
    except Exception as exc:
        logger.warning("Query analysis node error: %s", exc)
        state["complexity"] = "single_hop"
        state["freshness_required"] = False
        state["sub_questions"] = []
        state["query_confidence"] = 0.5

    return state


# ── Node 4: Static KB Retrieval ───────────────────────────────────────────────
async def static_retrieval_node(state: GraphState) -> GraphState:
    """Retrieve from the knowledge base using hybrid search."""
    query = state.get("enriched_query") or state["query"]
    user_id = state["user_id"]

    try:
        from app.retrieval.static.hybrid_retriever import hybrid_search
        chunks = await hybrid_search(query=query, user_id=user_id)
        state["kb_evidence"] = chunks
    except Exception as exc:
        logger.warning("Static retrieval node error: %s", exc)
        state["kb_evidence"] = []

    return state


# ── Node 5: Retrieval Evaluator ───────────────────────────────────────────────
async def retrieval_evaluator_node(state: GraphState) -> GraphState:
    """Grade whether KB evidence is correct, ambiguous, or incorrect."""
    query = state.get("enriched_query") or state["query"]
    kb_evidence = state.get("kb_evidence", [])

    try:
        from app.retrieval.evaluator.retrieval_grader import grade_retrieval
        grade = await grade_retrieval(query=query, retrieved_chunks=kb_evidence)
        state["retrieval_verdict"] = grade.verdict
    except Exception as exc:
        logger.warning("Retrieval evaluator node error: %s", exc)
        state["retrieval_verdict"] = "ambiguous"

    return state


# ── Node 6: Web Search ────────────────────────────────────────────────────────
async def web_search_node(state: GraphState) -> GraphState:
    """Execute web search and collect evidence."""
    query = state.get("enriched_query") or state["query"]

    try:
        from app.retrieval.web.search_agent import search_web, generate_search_queries
        search_queries = await generate_search_queries(query)
        all_web_evidence: List[dict] = []
        seen_urls = set()

        for sq in search_queries[:2]:  # Max 2 search rounds
            results = await search_web(sq)
            for r in results:
                url = r.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_web_evidence.append(r)

        state["web_evidence"] = all_web_evidence
    except Exception as exc:
        logger.warning("Web search node error: %s", exc)
        state["web_evidence"] = []

    return state


# ── Node 7: Evidence Fusion ───────────────────────────────────────────────────
async def evidence_fusion_node(state: GraphState) -> GraphState:
    """Merge KB and web evidence, score, resolve conflicts."""
    kb_evidence = state.get("kb_evidence", [])
    web_evidence = state.get("web_evidence", [])
    freshness_required = state.get("freshness_required", False)

    # Filter KB evidence if verdict is incorrect
    verdict = state.get("retrieval_verdict", "correct")
    if verdict == "incorrect":
        kb_evidence = []

    try:
        from app.fusion.evidence_merger import merge_evidence, resolve_conflicts
        all_evidence = merge_evidence(kb_evidence, web_evidence, freshness_required)
        conflict_report = resolve_conflicts(all_evidence, state["query"])
        state["all_evidence"] = all_evidence
        state["conflict_report"] = conflict_report
    except Exception as exc:
        logger.warning("Evidence fusion node error: %s", exc)
        state["all_evidence"] = kb_evidence + web_evidence
        state["conflict_report"] = None

    return state


# ── Node 8: Answer Synthesis ──────────────────────────────────────────────────
async def answer_synthesis_node(state: GraphState) -> GraphState:
    """Generate grounded answer from fused evidence."""
    query = state.get("enriched_query") or state["query"]
    all_evidence = state.get("all_evidence", [])
    memories_used = state.get("memories_used", [])
    route = _determine_route(state)

    try:
        from app.synthesis.answer_generator import synthesize_answer
        result = await synthesize_answer(
            query=query,
            evidence_items=all_evidence,
            memory_context=memories_used,
            route=route,
            conflict_report=state.get("conflict_report"),
        )
        state.update(result)
        state["route"] = route
    except Exception as exc:
        logger.warning("Answer synthesis node error: %s", exc)
        state["answer"] = "I encountered an error generating an answer. Please try again."
        state["route"] = "error"
        state["confidence"] = 0.0
        state["citations"] = []
        state["sources_used"] = []
        state["limitations"] = [str(exc)]
        state["verification_status"] = "error"
        state["retrieval_metadata"] = {}

    return state


# ── Node 9: Faithfulness Verification ────────────────────────────────────────
async def faithfulness_check_node(state: GraphState) -> GraphState:
    """Verify answer faithfulness against evidence."""
    answer = state.get("answer", "")
    all_evidence = state.get("all_evidence", [])

    if not answer or not all_evidence:
        state["verification_status"] = "skipped_no_evidence"
        return state

    try:
        # Simple heuristic: check if any citation markers are in the answer
        import re
        citations_found = bool(re.search(r"\[\d+\]", answer))
        if citations_found or len(all_evidence) == 0:
            state["verification_status"] = "passed"
        else:
            # Answer has no citations but we have evidence - flag it
            state["verification_status"] = "ungrounded"
            state["limitations"] = state.get("limitations", []) + [
                "Answer may not be fully grounded in retrieved evidence."
            ]
    except Exception as exc:
        logger.warning("Verification node error: %s", exc)
        state["verification_status"] = "error"

    return state


# ── Node 10: No-Retrieval Direct Answer ──────────────────────────────────────
async def direct_answer_node(state: GraphState) -> GraphState:
    """Answer directly from LLM knowledge without retrieval."""
    query = state.get("enriched_query") or state["query"]
    user_id = state.get("user_id", "")

    # Check if user has uploaded documents to provide helpful awareness
    doc_context = ""
    try:
        from app.database.mongodb import mongodb_manager
        docs = await mongodb_manager.list_documents(user_id)
        if not docs:
            import os, sqlite3
            for path in ["data/freshweb.db", "Freshweb_RAG/data/freshweb.db"]:
                if os.path.exists(path):
                    conn = sqlite3.connect(path)
                    cur = conn.cursor()
                    rows = cur.execute("SELECT filename, chunk_count, status FROM documents WHERE user_id = ?", (user_id,)).fetchall()
                    conn.close()
                    if rows:
                        docs = [{"filename": r[0], "chunk_count": r[1], "status": r[2]} for r in rows]
                        break
        if docs:
            names = [f"'{d.get('filename')}' ({d.get('chunk_count', 0)} chunks, status: {d.get('status')})" for d in docs]
            doc_context = f"\nUser's uploaded knowledge base documents: {', '.join(names)}.\n"
    except Exception as exc:
        logger.debug("Could not fetch doc context in direct_answer_node: %s", exc)

    try:
        from app.utils.llm_client import get_llm_client
        llm = get_llm_client()
        prompt = (
            "You are FreshWeb-RAG, an intelligent AI assistant.\n"
            f"{doc_context}"
            "Answer the question clearly and informatively. If the user asks about their knowledge base or documents, acknowledge their uploaded files listed above:\n\n"
            f"Question: {query}"
        )
        response = await llm.ainvoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        if isinstance(raw, list):
            raw = "".join(item["text"] if isinstance(item, dict) and "text" in item else str(item) for item in raw)
        answer = str(raw)
        state["answer"] = answer
        state["route"] = "no_retrieval"
        state["confidence"] = 0.7
        state["citations"] = []
        state["sources_used"] = []
        state["memories_used"] = state.get("memories_used", [])
        state["limitations"] = ["Answer based on direct synthesis; no external documents retrieved."]
        state["verification_status"] = "not_applicable"
        state["retrieval_metadata"] = {}
        state["all_evidence"] = []
    except Exception as exc:
        logger.warning("Direct answer node error: %s", exc)
        state["answer"] = (
            f"Regarding your query '{query}':\n\n"
            "This question was handled via direct synthesis. "
            "You can upload specific documents to your Knowledge Base in the sidebar to enable deep evidence-grounded retrieval."
        )
        state["route"] = "direct_answer"
        state["confidence"] = 0.6

    return state


# ── Node 11: Async Memory Persistence ────────────────────────────────────────
async def memory_update_node(state: GraphState) -> GraphState:
    """Extract and store memories from this interaction (non-blocking)."""
    # This runs after the response is sent; errors here don't affect the response
    try:
        conversation_id = state.get("conversation_id", "")
        query = state.get("query", "")
        answer = state.get("answer", "")
        user_id = state.get("user_id", "")

        if query and answer:
            from app.memory.memory_extractor import extract_memories_from_conversation
            from app.memory.memory_store import store_memory

            history = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer},
            ]
            memories = await extract_memories_from_conversation(history, conversation_id)

            for mem in memories:
                if mem.importance_score >= 0.5:
                    await store_memory(
                        user_id=user_id,
                        memory_type=mem.memory_type,
                        content=mem.content,
                        importance_score=mem.importance_score,
                        source_conversation_id=conversation_id,
                        provenance=mem.provenance,
                    )
    except Exception as exc:
        logger.warning("Memory update node error (non-critical): %s", exc)

    return state


def _determine_route(state: GraphState) -> str:
    """Determine the route label for analytics and response metadata."""
    complexity = state.get("complexity", "single_hop")
    freshness = state.get("freshness_required", False)
    verdict = state.get("retrieval_verdict", "correct")
    force_web = state.get("force_web_search", False)

    if complexity == "no_retrieval":
        return "no_retrieval"
    if force_web:
        return "web_forced"
    if freshness and not state.get("kb_evidence"):
        return "web_only"
    if verdict == "incorrect":
        return "web_corrective"
    if verdict == "ambiguous":
        return "hybrid"
    if complexity == "multi_hop":
        return "multi_hop"
    if state.get("web_evidence") and state.get("kb_evidence"):
        return "hybrid"
    if state.get("web_evidence"):
        return "web_supplemented"
    return "static_kb"

