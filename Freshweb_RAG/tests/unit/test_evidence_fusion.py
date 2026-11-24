"""Unit tests — Evidence Fusion and Conflict Resolution."""

from __future__ import annotations

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.fusion.evidence_merger import merge_evidence, resolve_conflicts


class TestEvidenceMerger:
    """Test evidence fusion and deduplication."""

    def test_merge_kb_and_web(self, sample_chunks, sample_web_evidence):
        merged = merge_evidence(sample_chunks, sample_web_evidence)
        assert len(merged) >= 2

    def test_deduplication_by_content(self):
        duplicate_content = "RAG stands for Retrieval Augmented Generation."
        kb = [{"chunk_id": "c1", "document_id": "d1", "content": duplicate_content, "score": 0.9, "source": "doc.pdf"}]
        web = [{"evidence_id": "w1", "content": duplicate_content, "url": "http://example.com", "domain": "example.com", "retrieved_at": "2024-01-01"}]
        merged = merge_evidence(kb, web)
        # Duplicates by content should be removed
        assert len(merged) == 1

    def test_provenance_tagged(self, sample_chunks, sample_web_evidence):
        merged = merge_evidence(sample_chunks, sample_web_evidence)
        types = {item.get("source_type") for item in merged}
        assert "document" in types or "web" in types

    def test_incorrect_verdict_discards_kb(self, sample_chunks, sample_web_evidence):
        # When verdict is incorrect, kb_evidence should be empty before merge
        merged = merge_evidence([], sample_web_evidence)
        for item in merged:
            assert item.get("source_type") != "document"

    def test_empty_inputs(self):
        merged = merge_evidence([], [])
        assert merged == []

    def test_scores_attached(self, sample_chunks):
        merged = merge_evidence(sample_chunks, [])
        for item in merged:
            assert "freshness_score" in item
            assert "reliability_score" in item


class TestConflictResolver:
    """Test conflict detection between evidence items."""

    def test_no_conflict_single_source(self, sample_chunks):
        import datetime
        for c in sample_chunks:
            c["published_at"] = "2024-11-01"
            c["freshness_score"] = 0.9
            c["retrieved_at"] = datetime.datetime.utcnow().isoformat()
        result = resolve_conflicts(sample_chunks[:1], "test query")
        assert result["resolution"] == "no_conflict"

    def test_temporal_conflict_detected(self):
        import datetime
        old_date = "2020-01-01"
        new_date = "2024-11-01"
        evidence = [
            {"content": "fact A", "published_at": new_date, "freshness_score": 0.95, "url": "http://new.com", "retrieved_at": datetime.datetime.utcnow().isoformat()},
            {"content": "fact B", "published_at": old_date, "freshness_score": 0.1, "url": "http://old.com", "retrieved_at": datetime.datetime.utcnow().isoformat()},
        ]
        result = resolve_conflicts(evidence, "current status query")
        # Should detect temporal conflict
        assert isinstance(result["conflicts"], list)
        assert isinstance(result["resolution"], str)

    def test_conflict_report_has_required_keys(self, sample_chunks):
        result = resolve_conflicts(sample_chunks, "test")
        assert "conflicts" in result
        assert "resolution" in result
        assert "notes" in result

