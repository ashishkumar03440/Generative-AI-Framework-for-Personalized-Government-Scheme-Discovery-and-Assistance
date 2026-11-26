"""Web search agent using Tavily."""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import List, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import settings

logger = logging.getLogger("freshweb_rag")

BLOCKED_DOMAINS = {"localhost", "127.0.0.1", "0.0.0.0", "169.254.0.0"}


def _is_safe_url(url: str) -> bool:
    """SSRF protection: reject internal/private URLs."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if host in BLOCKED_DOMAINS:
            return False
        if host.startswith("192.168.") or host.startswith("10.") or host.startswith("172."):
            return False
        if parsed.scheme not in ("http", "https"):
            return False
        return True
    except Exception:
        return False


class WebEvidence:
    def __init__(
        self,
        evidence_id: str,
        content: str,
        url: str,
        title: str,
        domain: str,
        published_at: Optional[str],
        retrieved_at: str,
        source_type: str = "web",
    ):
        self.evidence_id = evidence_id
        self.content = content
        self.url = url
        self.title = title
        self.domain = domain
        self.published_at = published_at
        self.retrieved_at = retrieved_at
        self.source_type = source_type

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "content": self.content,
            "url": self.url,
            "title": self.title,
            "domain": self.domain,
            "published_at": self.published_at,
            "retrieved_at": self.retrieved_at,
            "source_type": self.source_type,
        }


async def search_web(query: str, max_results: int = None) -> List[dict]:
    """
    Execute Tavily search and return structured web evidence.
    Falls back to empty list if Tavily is unavailable.
    """
    _max = max_results or settings.max_web_sources

    if not settings.tavily_api_key:
        logger.warning("TAVILY_API_KEY not configured. Web search unavailable.")
        return []

    try:
        return await _tavily_search(query, _max)
    except Exception as exc:
        logger.error("Web search failed: %s", exc)
        return []


async def _tavily_search(query: str, max_results: int) -> List[dict]:
    """Call Tavily Search API."""
    try:
        from tavily import AsyncTavilyClient
        client = AsyncTavilyClient(api_key=settings.tavily_api_key)
        response = await client.search(
            query=query,
            max_results=max_results,
            include_answer=False,
            include_raw_content=True,
        )
    except ImportError:
        # Fallback: direct HTTP call
        return await _tavily_http_search(query, max_results)

    results = response.get("results", [])
    retrieved_at = datetime.datetime.utcnow().isoformat()
    evidence_list = []

    seen_urls = set()
    for r in results:
        url = r.get("url", "")
        if url in seen_urls or not _is_safe_url(url):
            continue
        seen_urls.add(url)

        content = r.get("raw_content") or r.get("content", "")
        # Limit content size; treat as untrusted text
        content = content[:8000] if content else ""

        evidence_list.append(
            WebEvidence(
                evidence_id=str(uuid.uuid4()),
                content=content,
                url=url,
                title=r.get("title", ""),
                domain=urlparse(url).netloc,
                published_at=r.get("published_date"),
                retrieved_at=retrieved_at,
            ).to_dict()
        )

    return evidence_list


async def _tavily_http_search(query: str, max_results: int) -> List[dict]:
    """Direct HTTP fallback for Tavily API."""
    retrieved_at = datetime.datetime.utcnow().isoformat()
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "include_raw_content": True,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results", [])
    seen_urls = set()
    evidence_list = []

    for r in results:
        url = r.get("url", "")
        if url in seen_urls or not _is_safe_url(url):
            continue
        seen_urls.add(url)
        content = r.get("raw_content") or r.get("content", "")
        content = content[:8000] if content else ""
        evidence_list.append(
            WebEvidence(
                evidence_id=str(uuid.uuid4()),
                content=content,
                url=url,
                title=r.get("title", ""),
                domain=urlparse(url).netloc,
                published_at=r.get("published_date"),
                retrieved_at=retrieved_at,
            ).to_dict()
        )

    return evidence_list


async def generate_search_queries(original_query: str, context: str = "") -> List[str]:
    """Generate 1-3 targeted search queries for the original question."""
    try:
        from app.utils.llm_client import get_llm_client
        import re, json

        llm = get_llm_client()
        prompt = f"""Generate 1-3 targeted web search queries to answer this question. 
Focus on finding authoritative, recent sources.

Question: {original_query}
{f"Context: {context}" if context else ""}

Respond ONLY with a JSON array of query strings:
["query 1", "query 2", ...]"""

        from app.utils.llm_client import get_content_text
        response = await llm.ainvoke(prompt)
        text = get_content_text(response)
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            queries = json.loads(match.group())
            return [str(q) for q in queries[:3] if q]
    except Exception as exc:
        logger.warning("Query generation failed: %s", exc)

    # Fallback
    return [original_query]

