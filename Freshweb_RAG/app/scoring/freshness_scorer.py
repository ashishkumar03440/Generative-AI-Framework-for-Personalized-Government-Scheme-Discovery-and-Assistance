"""Freshness and reliability scoring engines."""

from __future__ import annotations

import datetime
import logging
import math
from typing import List, Optional

logger = logging.getLogger("freshweb_rag")

# Freshness decay constants by category (days)
DECAY_WINDOWS = {
    "stock_price": 1,
    "news": 7,
    "software_version": 90,
    "scientific": 365 * 3,
    "general": 180,
}

AUTHORITY_SCORES = {
    "arxiv.org": 0.9,
    "wikipedia.org": 0.75,
    "github.com": 0.80,
    "scholar.google.com": 0.90,
    "nature.com": 0.95,
    "ieee.org": 0.90,
    "docs.python.org": 0.95,
}


def score_freshness(
    published_at: Optional[str],
    last_modified: Optional[str],
    retrieved_at: str,
    category: str = "general",
) -> dict:
    """
    Score how fresh a piece of evidence is.

    Returns:
        freshness_score: 0.0 (stale) to 1.0 (very fresh)
        age_days: estimated age in days
        decay_category: category used for decay
        explanation: scoring explanation
    """
    decay_window = DECAY_WINDOWS.get(category, DECAY_WINDOWS["general"])

    # If neither publication nor modification date is available
    if not published_at and not last_modified:
        return {
            "freshness_score": 0.5,
            "age_days": None,
            "decay_category": category,
            "explanation": "Publication date unknown; moderate freshness assumed.",
        }

    date_str = published_at or last_modified
    age_days = 0.0
    try:
        # Try ISO format
        if date_str:
            # Strip timezone info for simple parsing
            clean = date_str.replace("Z", "+00:00")
            dt = datetime.datetime.fromisoformat(clean).replace(tzinfo=None)
            now = datetime.datetime.utcnow()
            age_days = max(0.0, (now - dt).total_seconds() / 86400)
    except Exception:
        # Unknown date — apply moderate penalty
        return {
            "freshness_score": 0.5,
            "age_days": None,
            "decay_category": category,
            "explanation": "Publication date unknown; moderate freshness assumed.",
        }

    # Exponential decay: score = e^(-age / decay_window)
    score = math.exp(-age_days / decay_window)
    score = round(min(max(score, 0.0), 1.0), 4)

    return {
        "freshness_score": score,
        "age_days": round(age_days, 1),
        "decay_category": category,
        "explanation": f"Evidence is {age_days:.0f} days old (decay window: {decay_window} days).",
    }


def score_reliability(
    domain: str,
    content: str,
    num_corroborating_sources: int = 1,
    relevance_score: float = 0.7,
) -> dict:
    """
    Score source reliability.

    Returns:
        reliability_score: 0.0 (unreliable) to 1.0 (highly reliable)
        breakdown: dict of contributing factors
    """
    # 1. Domain authority
    authority = AUTHORITY_SCORES.get(domain, 0.5)
    for known_domain, auth_score in AUTHORITY_SCORES.items():
        if domain.endswith(known_domain):
            authority = auth_score
            break

    # 2. Content specificity (penalize very short content)
    specificity = min(1.0, len(content.split()) / 150) if content else 0.0
    specificity = round(specificity, 3)

    # 3. Corroboration
    corroboration = min(1.0, 0.5 + (num_corroborating_sources - 1) * 0.15)

    # Weighted composite
    reliability = (
        authority * 0.40
        + relevance_score * 0.30
        + specificity * 0.15
        + corroboration * 0.15
    )
    reliability = round(min(max(reliability, 0.0), 1.0), 4)

    return {
        "reliability_score": reliability,
        "breakdown": {
            "domain_authority": authority,
            "content_specificity": specificity,
            "corroboration": corroboration,
            "relevance": relevance_score,
        },
    }


def score_evidence_list(evidence_items: List[dict], query_freshness_required: bool = False) -> List[dict]:
    """
    Score all evidence items and attach scores.
    """
    for item in evidence_items:
        freshness = score_freshness(
            published_at=item.get("published_at"),
            last_modified=item.get("last_modified"),
            retrieved_at=item.get("retrieved_at", datetime.datetime.utcnow().isoformat()),
        )
        reliability = score_reliability(
            domain=item.get("domain", "unknown"),
            content=item.get("content", ""),
            relevance_score=item.get("relevance_score", 0.7),
        )
        item["freshness_score"] = freshness["freshness_score"]
        item["freshness_detail"] = freshness
        item["reliability_score"] = reliability["reliability_score"]
        item["reliability_detail"] = reliability

    # Sort: if freshness required, weight freshness more; else weight reliability more
    def sort_key(item: dict) -> float:
        f = item.get("freshness_score", 0.5)
        r = item.get("reliability_score", 0.5)
        if query_freshness_required:
            return 0.6 * f + 0.4 * r
        return 0.3 * f + 0.7 * r

    evidence_items.sort(key=sort_key, reverse=True)
    return evidence_items

