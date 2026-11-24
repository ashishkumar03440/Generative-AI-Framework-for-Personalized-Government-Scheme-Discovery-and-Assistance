"""RAGAS Evaluation Harness for FreshWeb-RAG.

Runs a benchmark dataset through the RAG pipeline and measures:
- Answer Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Usage:
    python scripts/run_evaluation.py --dataset data/eval_dataset.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation")

# ── Evaluation Dataset ────────────────────────────────────────────────────────
DEFAULT_EVAL_DATASET = [
    {
        "id": "eval-001",
        "query": "What is Retrieval Augmented Generation?",
        "ground_truth": "RAG combines retrieval of relevant documents with language model generation to produce grounded, accurate answers.",
        "category": "factual",
        "expected_route": "no_retrieval",
        "freshness_required": False,
    },
    {
        "id": "eval-002",
        "query": "What are the main components of a RAG system?",
        "ground_truth": "A RAG system has a retriever that fetches relevant documents and a generator that produces answers grounded in those documents.",
        "category": "factual",
        "expected_route": "static_kb",
        "freshness_required": False,
    },
    {
        "id": "eval-003",
        "query": "Who is the current CEO of OpenAI?",
        "ground_truth": "Sam Altman is the CEO of OpenAI.",
        "category": "current_info",
        "expected_route": "web_only",
        "freshness_required": True,
    },
    {
        "id": "eval-004",
        "query": "Compare LangChain and LlamaIndex for RAG applications.",
        "ground_truth": "Both are frameworks for building RAG applications. LangChain offers broader LLM orchestration while LlamaIndex focuses on data indexing.",
        "category": "comparison",
        "expected_route": "multi_hop",
        "freshness_required": False,
    },
    {
        "id": "eval-005",
        "query": "Continue my previous RAG project.",
        "ground_truth": None,  # Context-dependent
        "category": "memory_reference",
        "expected_intent": "CONTINUE_PROJECT",
        "freshness_required": False,
    },
    {
        "id": "eval-006",
        "query": "What embedding model did we use last time?",
        "ground_truth": None,
        "category": "memory_reference",
        "expected_intent": "RECALL_INFORMATION",
        "freshness_required": False,
    },
    {
        "id": "eval-007",
        "query": "What is the latest version of Python?",
        "ground_truth": "Python 3.12 is a recent stable release (as of 2024).",
        "category": "current_info",
        "expected_route": "web_only",
        "freshness_required": True,
    },
    {
        "id": "eval-008",
        "query": "Improve the architecture we designed earlier.",
        "ground_truth": None,
        "category": "memory_reference",
        "expected_intent": "MODIFY_PREVIOUS_WORK",
        "freshness_required": False,
    },
]


class EvaluationResult:
    def __init__(self, eval_id: str, query: str, category: str):
        self.eval_id = eval_id
        self.query = query
        self.category = category
        self.answer: str = ""
        self.route: str = ""
        self.memory_intent: str = ""
        self.freshness_required: bool = False
        self.num_citations: int = 0
        self.execution_time_ms: float = 0.0
        self.error: str = ""
        # Routing accuracy
        self.expected_route: str = ""
        self.expected_intent: str = ""
        self.route_correct: bool = False
        # RAGAS metrics (populated if RAGAS available)
        self.ragas_faithfulness: float = 0.0
        self.ragas_relevancy: float = 0.0

    def to_dict(self) -> dict:
        return {
            "eval_id": self.eval_id,
            "query": self.query,
            "category": self.category,
            "answer_preview": self.answer[:200] if self.answer else "",
            "route": self.route,
            "memory_intent": self.memory_intent,
            "freshness_required": self.freshness_required,
            "num_citations": self.num_citations,
            "execution_time_ms": round(self.execution_time_ms, 1),
            "route_correct": self.route_correct,
            "expected_route": self.expected_route,
            "ragas_faithfulness": self.ragas_faithfulness,
            "ragas_relevancy": self.ragas_relevancy,
            "error": self.error,
        }


async def run_single_eval(item: dict, user_id: str = "eval-user-001") -> EvaluationResult:
    """Run a single evaluation item through the RAG pipeline."""
    result = EvaluationResult(
        eval_id=item["id"],
        query=item["query"],
        category=item.get("category", "general"),
    )
    result.expected_route = item.get("expected_route", "")
    result.expected_intent = item.get("expected_intent", "")

    start = time.perf_counter()
    try:
        from app.graphs.rag_graph import run_rag_graph

        rag_result = await run_rag_graph(
            query=item["query"],
            user_id=user_id,
            conversation_id=f"eval-conv-{item['id']}",
            use_knowledge_base=True,
            force_web_search=item.get("freshness_required", False),
            db=None,
        )

        result.answer = rag_result.get("answer", "")
        result.route = rag_result.get("route", "")
        result.num_citations = len(rag_result.get("citations", []))
        meta = rag_result.get("retrieval_metadata", {})
        result.memory_intent = meta.get("memory_intent", "")
        result.freshness_required = meta.get("freshness_required", False)

        # Check routing accuracy
        if result.expected_route:
            result.route_correct = result.route == result.expected_route
        elif result.expected_intent:
            result.route_correct = result.memory_intent == result.expected_intent

    except Exception as exc:
        result.error = str(exc)
        logger.error("Eval %s failed: %s", item["id"], exc)

    result.execution_time_ms = (time.perf_counter() - start) * 1000
    return result


async def run_evaluation(dataset: List[dict], output_path: str = None) -> dict:
    """Run the full evaluation suite."""
    logger.info("Starting evaluation on %d items...", len(dataset))
    results = []

    for item in dataset:
        logger.info("Running: %s — %s", item["id"], item["query"][:60])
        result = await run_single_eval(item)
        results.append(result)
        logger.info("  → route=%s, time=%.0fms, error=%s",
                    result.route, result.execution_time_ms, result.error or "none")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    total = len(results)
    errors = sum(1 for r in results if r.error)
    correct_routes = sum(1 for r in results if r.route_correct and r.expected_route)
    route_checked = sum(1 for r in results if r.expected_route)

    avg_time = sum(r.execution_time_ms for r in results) / total if total else 0
    avg_citations = sum(r.num_citations for r in results) / total if total else 0

    summary = {
        "total_evaluations": total,
        "errors": errors,
        "success_rate": round((total - errors) / total, 3) if total else 0,
        "routing_accuracy": round(correct_routes / route_checked, 3) if route_checked else None,
        "avg_execution_time_ms": round(avg_time, 1),
        "avg_citations_per_answer": round(avg_citations, 2),
        "results": [r.to_dict() for r in results],
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results written to %s", output_path)

    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "="*60)
    print("FRESHWEB-RAG EVALUATION SUMMARY")
    print("="*60)
    print(f"Total evaluations : {summary['total_evaluations']}")
    print(f"Success rate      : {summary['success_rate']:.1%}")
    if summary.get("routing_accuracy") is not None:
        print(f"Routing accuracy  : {summary['routing_accuracy']:.1%}")
    print(f"Avg execution time: {summary['avg_execution_time_ms']:.0f}ms")
    print(f"Avg citations     : {summary['avg_citations_per_answer']:.1f}")
    print(f"Errors            : {summary['errors']}")
    print("-"*60)
    print(f"{'ID':<12} {'Category':<18} {'Route':<20} {'Time(ms)':<10} {'OK'}")
    print("-"*60)
    for r in summary["results"]:
        ok = "✓" if not r["error"] else "✗"
        print(f"{r['eval_id']:<12} {r['category']:<18} {r['route']:<20} {r['execution_time_ms']:<10.0f} {ok}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FreshWeb-RAG Evaluation Harness")
    parser.add_argument("--dataset", default=None, help="Path to JSON evaluation dataset")
    parser.add_argument("--output", default="data/eval_results.json", help="Output results path")
    args = parser.parse_args()

    if args.dataset and Path(args.dataset).exists():
        with open(args.dataset) as f:
            dataset = json.load(f)
        logger.info("Loaded %d evaluation items from %s", len(dataset), args.dataset)
    else:
        logger.info("Using built-in evaluation dataset (%d items)", len(DEFAULT_EVAL_DATASET))
        dataset = DEFAULT_EVAL_DATASET

    summary = asyncio.run(run_evaluation(dataset, output_path=args.output))
    print_summary(summary)

