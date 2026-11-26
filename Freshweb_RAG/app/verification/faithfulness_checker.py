"""Faithfulness checker and claim verifier.

Verifies that the generated answer is grounded in retrieved evidence.
Extracts factual claims, maps them to evidence, and flags unsupported statements.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

logger = logging.getLogger("freshweb_rag")


class VerificationResult:
    def __init__(
        self,
        status: str,
        supported_claims: List[str],
        unsupported_claims: List[str],
        invalid_citations: List[str],
        corrective_actions: List[str],
    ):
        self.status = status  # passed | failed | partial | skipped
        self.supported_claims = supported_claims
        self.unsupported_claims = unsupported_claims
        self.invalid_citations = invalid_citations
        self.corrective_actions = corrective_actions

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "invalid_citations": self.invalid_citations,
            "corrective_actions": self.corrective_actions,
        }


async def verify_answer(
    answer: str,
    evidence_items: List[dict],
    max_retries: int = 1,
) -> VerificationResult:
    """
    Verify faithfulness of the generated answer against evidence.

    Returns a VerificationResult with status and claim breakdown.
    """
    if not answer or not evidence_items:
        return VerificationResult(
            status="skipped",
            supported_claims=[],
            unsupported_claims=[],
            invalid_citations=[],
            corrective_actions=["No evidence available for verification."],
        )

    try:
        return await _llm_verify(answer, evidence_items)
    except Exception as exc:
        logger.warning("LLM verification failed: %s. Using heuristic.", exc)
        return _heuristic_verify(answer, evidence_items)


def _heuristic_verify(answer: str, evidence_items: List[dict]) -> VerificationResult:
    """
    Heuristic verification:
    - Check inline citations present
    - Check that cited content exists in evidence
    - Flag uncited factual sentences
    """
    # Find citation references in answer
    cited_numbers = set(re.findall(r"\[(\d+)\]", answer))
    max_citation = max((int(n) for n in cited_numbers), default=0)

    invalid_citations = []
    if max_citation > len(evidence_items):
        invalid_citations.append(
            f"Citation [{max_citation}] exceeds available evidence ({len(evidence_items)} items)."
        )

    # Split answer into sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    supported = []
    unsupported = []

    for sentence in sentences:
        # Skip meta-sentences about uncertainty
        if any(phrase in sentence.lower() for phrase in [
            "i don't know", "unclear", "uncertain", "insufficient",
            "i cannot", "based on", "according to", "note that",
        ]):
            supported.append(sentence)
            continue

        # Check if sentence has citation
        if re.search(r"\[\d+\]", sentence):
            supported.append(sentence)
        else:
            # Check if content exists in evidence (keyword overlap)
            sentence_words = set(re.findall(r"\w+", sentence.lower()))
            evidence_text = " ".join(e.get("content", "")[:200] for e in evidence_items).lower()
            evidence_words = set(re.findall(r"\w+", evidence_text))
            overlap = len(sentence_words & evidence_words)
            if overlap >= 3 or len(sentence_words) < 5:
                supported.append(sentence)
            else:
                unsupported.append(sentence)

    if not unsupported and not invalid_citations:
        status = "passed"
    elif len(unsupported) <= 1 and not invalid_citations:
        status = "partial"
    else:
        status = "failed"

    corrective_actions = []
    if unsupported:
        corrective_actions.append(
            f"{len(unsupported)} sentence(s) lack citation support — add inline citations or remove."
        )
    if invalid_citations:
        corrective_actions.extend(invalid_citations)

    return VerificationResult(
        status=status,
        supported_claims=supported[:5],  # Return sample only
        unsupported_claims=unsupported,
        invalid_citations=invalid_citations,
        corrective_actions=corrective_actions,
    )


async def _llm_verify(answer: str, evidence_items: List[dict]) -> VerificationResult:
    """LLM-based verification with structured output."""
    from app.utils.llm_client import get_llm_client
    import json

    llm = get_llm_client()
    evidence_block = "\n---\n".join(
        f"[{i+1}] {item.get('content', '')[:400]}"
        for i, item in enumerate(evidence_items[:5])
    )

    prompt = f"""You are a fact-checking system. Verify that this answer is grounded in the provided evidence.

Answer:
{answer[:1500]}

Evidence:
{evidence_block}

For each factual claim in the answer:
1. Check if it is supported by the evidence.
2. Check if citations (like [1], [2]) are accurate.
3. Flag unsupported claims or fake citations.

Respond ONLY with JSON:
{{
  "status": "passed" | "partial" | "failed",
  "supported_claims": ["<claim>", ...],
  "unsupported_claims": ["<claim>", ...],
  "invalid_citations": ["<description>", ...],
  "corrective_actions": ["<action>", ...]
}}

Rules:
- "passed": All factual claims are grounded and all citations valid.
- "partial": Minor unsupported claims, no fabricated citations.
- "failed": Multiple unsupported claims or fabricated citations detected."""

    from app.utils.llm_client import get_content_text
    response = await llm.ainvoke(prompt)
    text = get_content_text(response)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON in LLM verification response")
    data = json.loads(match.group())
    return VerificationResult(
        status=data.get("status", "partial"),
        supported_claims=data.get("supported_claims", []),
        unsupported_claims=data.get("unsupported_claims", []),
        invalid_citations=data.get("invalid_citations", []),
        corrective_actions=data.get("corrective_actions", []),
    )

