"""Complete LangGraph RAG orchestration graph."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graphs.graph_state import GraphState
from app.graphs.graph_nodes import (
    answer_synthesis_node,
    direct_answer_node,
    evidence_fusion_node,
    faithfulness_check_node,
    memory_aware_analysis_node,
    memory_retrieval_node,
    memory_update_node,
    query_analysis_node,
    retrieval_evaluator_node,
    static_retrieval_node,
    web_search_node,
)
from app.graphs.graph_edges import (
    route_after_faithfulness,
    route_after_memory_analysis,
    route_after_query_analysis,
    route_after_retrieval_eval,
)

logger = logging.getLogger("freshweb_rag")


def build_rag_graph() -> Any:
    """Build and compile the full LangGraph RAG workflow."""
    graph = StateGraph(GraphState)

    # ── Add Nodes ─────────────────────────────────────────────────────────────
    graph.add_node("memory_aware_analysis", memory_aware_analysis_node)
    graph.add_node("memory_retrieval", memory_retrieval_node)
    graph.add_node("query_analysis", query_analysis_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("static_retrieval", static_retrieval_node)
    graph.add_node("retrieval_evaluator", retrieval_evaluator_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("evidence_fusion", evidence_fusion_node)
    graph.add_node("answer_synthesis", answer_synthesis_node)
    graph.add_node("faithfulness_check", faithfulness_check_node)
    graph.add_node("memory_update", memory_update_node)

    # ── Entry Point ───────────────────────────────────────────────────────────
    graph.add_edge(START, "memory_aware_analysis")

    # ── Conditional: after memory-aware analysis ──────────────────────────────
    graph.add_conditional_edges(
        "memory_aware_analysis",
        route_after_memory_analysis,
        {
            "retrieve_memory": "memory_retrieval",
            "query_analysis": "query_analysis",
            "clarification": "direct_answer",  # simplified: return clarification request
        },
    )

    # Memory retrieval always leads to query analysis
    graph.add_edge("memory_retrieval", "query_analysis")

    # ── Conditional: after query analysis ────────────────────────────────────
    graph.add_conditional_edges(
        "query_analysis",
        route_after_query_analysis,
        {
            "direct_answer": "direct_answer",
            "web_search": "web_search",
            "static_retrieval": "static_retrieval",
        },
    )

    # Static retrieval → evaluator
    graph.add_edge("static_retrieval", "retrieval_evaluator")

    # ── Conditional: after retrieval evaluation ───────────────────────────────
    graph.add_conditional_edges(
        "retrieval_evaluator",
        route_after_retrieval_eval,
        {
            "evidence_fusion": "evidence_fusion",
            "web_search": "web_search",
        },
    )

    # Web search → evidence fusion
    graph.add_edge("web_search", "evidence_fusion")

    # Evidence fusion → answer synthesis
    graph.add_edge("evidence_fusion", "answer_synthesis")

    # Direct answer → faithfulness check (abbreviated)
    graph.add_edge("direct_answer", "faithfulness_check")

    # Answer synthesis → faithfulness check
    graph.add_edge("answer_synthesis", "faithfulness_check")

    # ── Conditional: after faithfulness check ─────────────────────────────────
    graph.add_conditional_edges(
        "faithfulness_check",
        route_after_faithfulness,
        {
            "memory_update": "memory_update",
            "re_synthesize": "answer_synthesis",  # one retry
        },
    )

    # Memory update → END
    graph.add_edge("memory_update", END)

    return graph.compile()


# Singleton compiled graph
_rag_graph = None


def get_rag_graph():
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = build_rag_graph()
    return _rag_graph


async def run_rag_graph(
    query: str,
    user_id: str,
    conversation_id: str,
    use_knowledge_base: bool = True,
    force_web_search: bool = False,
    db=None,
) -> dict:
    """
    Run the full RAG pipeline through LangGraph.

    Returns a dict with answer, citations, route, etc.
    """
    graph = get_rag_graph()

    initial_state: GraphState = {
        "query": query,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "use_knowledge_base": use_knowledge_base,
        "force_web_search": force_web_search,
        "iteration_count": 0,
        # Defaults
        "memories_used": [],
        "kb_evidence": [],
        "web_evidence": [],
        "all_evidence": [],
    }

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("RAG graph execution error: %s", exc)
        return {
            "answer": "An error occurred during processing. Please try again.",
            "route": "error",
            "confidence": 0.0,
            "citations": [],
            "sources_used": [],
            "memories_used": [],
            "limitations": [str(exc)],
            "verification_status": "error",
            "retrieval_metadata": {},
        }

    return {
        "answer": final_state.get("answer", ""),
        "route": final_state.get("route", "unknown"),
        "confidence": final_state.get("confidence", 0.0),
        "citations": final_state.get("citations", []),
        "sources_used": final_state.get("sources_used", []),
        "memories_used": final_state.get("memories_used", []),
        "limitations": final_state.get("limitations", []),
        "verification_status": final_state.get("verification_status", "unverified"),
        "retrieval_metadata": {
            "complexity": final_state.get("complexity"),
            "freshness_required": final_state.get("freshness_required"),
            "retrieval_verdict": final_state.get("retrieval_verdict"),
            "num_kb_chunks": len(final_state.get("kb_evidence", [])),
            "num_web_sources": len(final_state.get("web_evidence", [])),
            "memory_intent": final_state.get("memory_intent"),
        },
    }

