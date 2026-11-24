"""Evidence fusion and conflict resolution."""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger("freshweb_rag")


def merge_evidence(
    kb_evidence: List[dict],
    web_evidence: List[dict],
    query_freshness_required: bool = False,
) -> List[dict]:
    """
    Merge KB and web evidence with provenance tagging and deduplication.
    """
    merged = []

    for item in kb_evidence:
        item = dict(item)
        item["source_type"] = "document"
        item.setdefault("evidence_id", item.get("chunk_id", ""))
        merged.append(item)

    for item in web_evidence:
        item = dict(item)
        item["source_type"] = item.get("source_type", "web")
        merged.append(item)

    # Deduplicate by content hash (simple first-500-char fingerprint)
    seen = set()
    deduped = []
    for item in merged:
        fingerprint = (item.get("content", "")[:500]).strip().lower()
        if fingerprint not in seen:
            seen.add(fingerprint)
            deduped.append(item)

    # Score and sort
    from app.scoring.freshness_scorer import score_evidence_list
    scored = score_evidence_list(deduped, query_freshness_required)

    return scored


def resolve_conflicts(evidence_items: List[dict], query: str) -> dict:
    """
    Detect and report conflicting claims in evidence.

    Returns a conflict report with:
        - conflicts: list of conflicting claim pairs
        - resolution: "latest_wins" | "authoritative_source" | "uncertain"
        - notes: human-readable explanation
    """
    if len(evidence_items) < 2:
        return {"conflicts": [], "resolution": "no_conflict", "notes": "Single or no evidence."}

    # Simplified conflict detection: check if sources give radically different short answers
    # In production, use NLI-based entailment checking
    conflicts = []
    notes_parts = []

    # Check date conflicts for time-sensitive info
    dates = [
        (i, item.get("published_at"), item.get("freshness_score", 0.5))
        for i, item in enumerate(evidence_items)
        if item.get("published_at")
    ]

    if dates and len(dates) > 1:
        dates.sort(key=lambda x: x[2], reverse=True)
        newest_idx = dates[0][0]
        oldest_idx = dates[-1][0]
        if dates[0][2] - dates[-1][2] > 0.5:
            conflicts.append({
                "type": "temporal",
                "newer_source": evidence_items[newest_idx].get("url") or evidence_items[newest_idx].get("source"),
                "older_source": evidence_items[oldest_idx].get("url") or evidence_items[oldest_idx].get("source"),
                "description": "Evidence sources have significantly different dates. Newer source may supersede older.",
            })
            notes_parts.append("Temporal conflict detected; preferring more recent source.")

    resolution = "no_conflict"
    if conflicts:
        resolution = "latest_wins" if any(c["type"] == "temporal" for c in conflicts) else "uncertain"

    return {
        "conflicts": conflicts,
        "resolution": resolution,
        "notes": " ".join(notes_parts) if notes_parts else "No significant conflicts detected.",
    }

