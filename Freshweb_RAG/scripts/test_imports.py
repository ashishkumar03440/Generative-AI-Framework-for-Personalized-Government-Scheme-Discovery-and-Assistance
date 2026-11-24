"""
Test that all core modules import correctly without real service connections.
"""
import sys, os
sys.path.insert(0, r'D:\SEM7proj\Freshweb_RAG')
os.chdir(r'D:\SEM7proj\Freshweb_RAG')

errors = []

modules_to_test = [
    "app.config",
    "app.utils.exceptions",
    "app.utils.logging",
    "app.utils.llm_client",
    "app.utils.auth",
    "app.database.models",
    "app.database.session",
    "app.router.query_analyzer",
    "app.retrieval.static.vector_store",
    "app.retrieval.static.hybrid_retriever",
    "app.retrieval.static.document_processor",
    "app.retrieval.static.reranker",
    "app.retrieval.evaluator.retrieval_grader",
    "app.retrieval.web.search_agent",
    "app.scoring.freshness_scorer",
    "app.fusion.evidence_merger",
    "app.synthesis.answer_generator",
    "app.memory.memory_retriever",
    "app.memory.memory_store",
    "app.memory.memory_extractor",
    "app.memory_aware_router.router",
    "app.memory_aware_router.entity_resolver",
    "app.memory_aware_router.query_rewriter",
    "app.cache.redis_cache",
    "app.analytics.metrics",
    "app.verification.faithfulness_checker",
    "app.graphs.graph_state",
    "app.graphs.graph_edges",
    "app.graphs.graph_nodes",
    "app.graphs.rag_graph",
    "app.api.routes.health",
    "app.api.routes.chat",
    "app.api.routes.documents",
    "app.api.routes.memory",
    "app.api.routes.conversations",
    "app.api.routes.search",
    "app.api.routes.analytics",
    "app.api.routes.memory_aware",
    "app.main",
]

for mod in modules_to_test:
    try:
        __import__(mod)
        print(f"  OK  {mod}")
    except Exception as e:
        errors.append((mod, str(e)))
        print(f"  FAIL {mod}: {e}")

print()
if errors:
    print(f"FAILED: {len(errors)} modules")
    for mod, err in errors:
        print(f"  - {mod}: {err}")
    sys.exit(1)
else:
    print(f"All {len(modules_to_test)} modules imported successfully!")
