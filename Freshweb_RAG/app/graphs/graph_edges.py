"""LangGraph conditional edge functions."""

from app.graphs.graph_state import GraphState


def route_after_memory_analysis(state: GraphState) -> str:
    """Route based on memory-aware analysis result."""
    if state.get("requires_clarification"):
        return "clarification"
    if state.get("has_historical_reference"):
        return "retrieve_memory"
    return "query_analysis"


def route_after_query_analysis(state: GraphState) -> str:
    """Route based on query complexity and freshness."""
    complexity = state.get("complexity", "single_hop")
    freshness = state.get("freshness_required", False)
    force_web = state.get("force_web_search", False)
    use_kb = state.get("use_knowledge_base", True)

    if complexity == "no_retrieval" and not force_web and not freshness:
        return "direct_answer"

    if force_web or (freshness and not use_kb):
        return "web_search"

    if complexity == "multi_hop":
        return "static_retrieval"  # will also trigger web for multi-hop

    return "static_retrieval"


def route_after_retrieval_eval(state: GraphState) -> str:
    """Route based on retrieval quality verdict."""
    verdict = state.get("retrieval_verdict", "ambiguous")
    freshness = state.get("freshness_required", False)
    force_web = state.get("force_web_search", False)

    if verdict == "correct" and not freshness and not force_web:
        return "evidence_fusion"  # skip web, go straight to fusion

    # ambiguous or incorrect → supplement with web
    return "web_search"


def route_after_faithfulness(state: GraphState) -> str:
    """Route based on verification result."""
    status = state.get("verification_status", "passed")
    iterations = state.get("iteration_count", 0)

    if status in ("passed", "skipped_no_evidence", "not_applicable"):
        return "memory_update"

    if status in ("ungrounded", "failed") and iterations < 1:
        # Allow one regeneration attempt
        return "re_synthesize"

    return "memory_update"

