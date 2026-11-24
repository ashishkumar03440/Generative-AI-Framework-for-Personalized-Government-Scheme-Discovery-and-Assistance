"""Memory extractor — identifies memory-worthy content from conversations."""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger("freshweb_rag")


class ExtractedMemory:
    def __init__(self, memory_type: str, content: str, importance_score: float, provenance: str):
        self.memory_type = memory_type
        self.content = content
        self.importance_score = importance_score
        self.provenance = provenance


async def extract_memories_from_conversation(
    conversation_history: List[dict],
    conversation_id: str,
) -> List[ExtractedMemory]:
    """
    Extract memory-worthy facts from a conversation.

    Returns a list of ExtractedMemory objects.
    """
    if not conversation_history:
        return []

    try:
        return await _llm_extract(conversation_history, conversation_id)
    except Exception as exc:
        logger.warning("LLM memory extraction failed: %s", exc)
        return _rule_extract(conversation_history, conversation_id)


async def _llm_extract(
    history: List[dict],
    conversation_id: str,
) -> List[ExtractedMemory]:
    from app.utils.llm_client import get_llm_client
    import json

    llm = get_llm_client()

    # Format recent conversation
    formatted = "\n".join(
        f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')[:300]}"
        for msg in history[-10:]  # Last 10 messages
    )

    prompt = f"""Analyze this conversation and extract ONLY genuinely important, reusable information worth remembering.

Conversation:
{formatted}

Extract memories in these categories:
- "episodic": Specific events, decisions, problems solved, project milestones.
- "semantic": Technical preferences, configurations, tools chosen, project facts.
- "procedural": Workflows, approaches, instructions the user wants reused.

RULES:
- Only extract genuinely useful, non-obvious information.
- Do NOT store raw conversation text or trivial exchanges.
- Do NOT store passwords, API keys, or sensitive personal data.
- Assign importance_score 0.0-1.0 (1.0 = critical to remember).
- If nothing important, return empty list.

Respond ONLY with JSON array:
[
  {{
    "memory_type": "episodic" | "semantic" | "procedural",
    "content": "<concise memory description>",
    "importance_score": <0.0-1.0>
  }},
  ...
]"""

    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []

    items = json.loads(match.group())
    memories = []
    for item in items:
        if not item.get("content") or len(item["content"]) < 10:
            continue
        memories.append(ExtractedMemory(
            memory_type=item.get("memory_type", "episodic"),
            content=item["content"],
            importance_score=float(item.get("importance_score", 0.5)),
            provenance=f"conversation:{conversation_id}",
        ))
    return memories


def _rule_extract(history: List[dict], conversation_id: str) -> List[ExtractedMemory]:
    """Simple rule-based fallback extraction."""
    memories = []
    for msg in history:
        content = msg.get("content", "")
        # Look for explicit preferences or technical decisions
        if any(phrase in content.lower() for phrase in [
            "i prefer", "we decided", "use this", "my project",
            "configuration is", "the architecture", "we chose",
        ]):
            memories.append(ExtractedMemory(
                memory_type="semantic",
                content=content[:300],
                importance_score=0.5,
                provenance=f"conversation:{conversation_id}",
            ))
    return memories[:5]  # Limit

