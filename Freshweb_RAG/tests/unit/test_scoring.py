"""Unit tests — Freshness and Reliability Scoring."""

from __future__ import annotations

import datetime
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.scoring.freshness_scorer import score_freshness, score_reliability, score_evidence_list


class TestFreshnessScorer:
    """Test freshness decay scoring."""

    def test_very_fresh_today(self):
        today = datetime.datetime.utcnow().isoformat()
        result = score_freshness(published_at=today, last_modified=None, retrieved_at=today)
        assert result["freshness_score"] > 0.95

    def test_stale_one_year_ago(self):
        one_year_ago = (datetime.datetime.utcnow() - datetime.timedelta(days=365)).isoformat()
        today = datetime.datetime.utcnow().isoformat()
        result = score_freshness(published_at=one_year_ago, last_modified=None, retrieved_at=today)
        assert result["freshness_score"] < 0.5

    def test_unknown_date_moderate_score(self):
        today = datetime.datetime.utcnow().isoformat()
        result = score_freshness(published_at=None, last_modified=None, retrieved_at=today)
        assert result["freshness_score"] == 0.5
        assert "unknown" in result["explanation"].lower()

    def test_stock_price_very_short_window(self):
        yesterday = (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat()
        today = datetime.datetime.utcnow().isoformat()
        result = score_freshness(published_at=yesterday, last_modified=None, retrieved_at=today, category="stock_price")
        # Stock prices decay very fast — 1 day old should be significantly decayed
        assert result["freshness_score"] < 0.5

    def test_scientific_stable_long_window(self):
        one_year_ago = (datetime.datetime.utcnow() - datetime.timedelta(days=365)).isoformat()
        today = datetime.datetime.utcnow().isoformat()
        result = score_freshness(published_at=one_year_ago, last_modified=None, retrieved_at=today, category="scientific")
        # Scientific facts decay slowly — 1 year should still be fairly fresh
        assert result["freshness_score"] > 0.5

    def test_score_range(self):
        today = datetime.datetime.utcnow().isoformat()
        result = score_freshness(published_at=today, last_modified=None, retrieved_at=today)
        assert 0.0 <= result["freshness_score"] <= 1.0

    def test_age_days_returned(self):
        past = (datetime.datetime.utcnow() - datetime.timedelta(days=30)).isoformat()
        today = datetime.datetime.utcnow().isoformat()
        result = score_freshness(published_at=past, last_modified=None, retrieved_at=today)
        assert result["age_days"] is not None
        assert 29 <= result["age_days"] <= 31


class TestReliabilityScorer:
    """Test source reliability scoring."""

    def test_known_authoritative_domain(self):
        result = score_reliability(
            domain="arxiv.org",
            content="This paper presents a novel approach to RAG." * 20,
            relevance_score=0.9,
        )
        assert result["reliability_score"] > 0.7

    def test_unknown_domain_medium_score(self):
        result = score_reliability(
            domain="random-blog.io",
            content="Some content about AI." * 10,
            relevance_score=0.5,
        )
        assert 0.2 < result["reliability_score"] < 0.8

    def test_very_short_content_low_specificity(self):
        result = score_reliability(
            domain="example.com",
            content="Short.",
            relevance_score=0.5,
        )
        assert result["breakdown"]["content_specificity"] < 0.3

    def test_corroboration_increases_score(self):
        base = score_reliability("example.com", "content " * 50, relevance_score=0.5, num_corroborating_sources=1)
        corroborated = score_reliability("example.com", "content " * 50, relevance_score=0.5, num_corroborating_sources=4)
        assert corroborated["reliability_score"] > base["reliability_score"]

    def test_breakdown_keys_present(self):
        result = score_reliability("example.com", "some content", relevance_score=0.5)
        assert "domain_authority" in result["breakdown"]
        assert "content_specificity" in result["breakdown"]
        assert "corroboration" in result["breakdown"]

    def test_score_range(self):
        result = score_reliability("example.com", "test", relevance_score=0.5)
        assert 0.0 <= result["reliability_score"] <= 1.0


class TestEvidenceListScoring:
    """Test scoring and sorting of evidence lists."""

    def test_freshness_sorted_first_when_required(self):
        today = datetime.datetime.utcnow().isoformat()
        old = (datetime.datetime.utcnow() - datetime.timedelta(days=300)).isoformat()
        evidence = [
            {"content": "old", "domain": "example.com", "published_at": old, "retrieved_at": today},
            {"content": "fresh", "domain": "arxiv.org", "published_at": today, "retrieved_at": today},
        ]
        scored = score_evidence_list(evidence, query_freshness_required=True)
        # Fresh evidence should rank higher when freshness matters
        assert scored[0]["content"] == "fresh"

    def test_all_evidence_scored(self):
        today = datetime.datetime.utcnow().isoformat()
        evidence = [
            {"content": "item1", "domain": "arxiv.org", "published_at": today, "retrieved_at": today},
            {"content": "item2", "domain": "example.com", "published_at": today, "retrieved_at": today},
        ]
        scored = score_evidence_list(evidence)
        for item in scored:
            assert "freshness_score" in item
            assert "reliability_score" in item

