"""LangGraph state definitions."""

from __future__ import annotations

from typing import Any, List, Literal, Optional
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    # Input
    query: str
    user_id: str
    conversation_id: str
    use_knowledge_base: bool
    force_web_search: bool

    # Memory-aware analysis
    memory_intent: str
    has_historical_reference: bool
    reference_type: Optional[str]
    requires_clarification: bool
    enriched_query: str
    memories_used: List[dict]

    # Query analysis
    complexity: Literal["no_retrieval", "single_hop", "multi_hop"]
    freshness_required: bool
    sub_questions: List[str]
    query_confidence: float

    # Retrieval
    kb_evidence: List[dict]
    retrieval_verdict: str  # correct | ambiguous | incorrect
    web_evidence: List[dict]

    # Scoring & fusion
    all_evidence: List[dict]
    conflict_report: Optional[dict]

    # Output
    answer: str
    route: str
    citations: List[dict]
    sources_used: List[str]
    limitations: List[str]
    confidence: float
    verification_status: str
    retrieval_metadata: dict

    # Control
    error: Optional[str]
    iteration_count: int

