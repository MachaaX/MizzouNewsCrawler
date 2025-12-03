"""
Tests for database-driven content validation in extraction pipeline.

Tests the integration of BalancedBoundaryContentCleaner into extraction
workflow to validate articles have sufficient non-boilerplate content
before saving to the articles table.
"""

from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
from unittest.mock import Mock, patch

import pytest

import src.cli.commands.extraction as extraction


class TestContentValidationIntegration:
    """Test database-driven content validation in extraction command."""

    def test_content_cleaner_instantiated_in_handle_extraction_command(
        self, monkeypatch
    ):
        """Test that BalancedBoundaryContentCleaner is created with telemetry disabled."""
        instantiation_calls = []

        class FakeContentCleaner:
            def __init__(self, **kwargs):
                instantiation_calls.append(kwargs)
                self.enable_telemetry = kwargs.get("enable_telemetry", True)

        class FakeExtractor:
            def get_driver_stats(self):
                return {
                    "has_persistent_driver": False,
                    "driver_reuse_count": 0,
                    "driver_creation_count": 0,
                }

            def close_persistent_driver(self):
                pass

        class FakeTelemetry:
            def record_extraction(self, *args, **kwargs):
                pass

        def fake_process_batch(*args, **kwargs):
            return {
                "processed": 0,
                "domains_processed": [],
                "same_domain_consecutive": 0,
            }

        def fake_domain_analysis(args, session):
            return {
                "unique_domains": 0,
                "is_single_domain": False,
                "sample_domains": [],
            }

        monkeypatch.setattr(extraction, "ContentExtractor", FakeExtractor)
        monkeypatch.setattr(extraction, "BylineCleaner", lambda: object())
        monkeypatch.setattr(
            extraction, "BalancedBoundaryContentCleaner", FakeContentCleaner
        )
        monkeypatch.setattr(
            extraction, "ComprehensiveExtractionTelemetry", lambda: FakeTelemetry()
        )
        monkeypatch.setattr(extraction, "_process_batch", fake_process_batch)
        monkeypatch.setattr(
            extraction, "_analyze_dataset_domains", fake_domain_analysis
        )

        args = Namespace(
            batches=0, limit=1, source=None, dataset=None, exhaust_queue=False
        )
        extraction.handle_extraction_command(args)

        # Verify content cleaner was instantiated with telemetry disabled
        assert len(instantiation_calls) == 1
        assert instantiation_calls[0] == {"enable_telemetry": False}

    def test_content_cleaner_passed_to_process_batch(self, monkeypatch):
        """Test that content_cleaner is passed as parameter to _process_batch."""
        process_batch_calls = []

        class FakeContentCleaner:
            def __init__(self, **kwargs):
                self.enable_telemetry = False

        class FakeExtractor:
            def get_driver_stats(self):
                return {"has_persistent_driver": False}

            def close_persistent_driver(self):
                pass

        class FakeTelemetry:
            def record_extraction(self, *args, **kwargs):
                pass

        def fake_process_batch(
            args,
            extractor,
            byline_cleaner,
            content_cleaner,
            telemetry,
            per_batch,
            batch_num,
            host_403_tracker,
            domains_for_cleaning,
            **kwargs,
        ):
            process_batch_calls.append(
                {
                    "content_cleaner": content_cleaner,
                    "content_cleaner_type": type(content_cleaner).__name__,
                }
            )
            return {
                "processed": 0,
                "domains_processed": [],
                "same_domain_consecutive": 0,
            }

        def fake_domain_analysis(args, session):
            return {
                "unique_domains": 0,
                "is_single_domain": False,
                "sample_domains": [],
            }

        monkeypatch.setattr(extraction, "ContentExtractor", FakeExtractor)
        monkeypatch.setattr(extraction, "BylineCleaner", lambda: object())
        monkeypatch.setattr(
            extraction, "BalancedBoundaryContentCleaner", FakeContentCleaner
        )
        monkeypatch.setattr(
            extraction, "ComprehensiveExtractionTelemetry", lambda: FakeTelemetry()
        )
        monkeypatch.setattr(extraction, "_process_batch", fake_process_batch)
        monkeypatch.setattr(
            extraction, "_analyze_dataset_domains", fake_domain_analysis
        )

        args = Namespace(
            batches=1, limit=1, source=None, dataset=None, exhaust_queue=False
        )
        extraction.handle_extraction_command(args)

        # Verify content_cleaner was passed to process_batch
        assert len(process_batch_calls) >= 1
        assert process_batch_calls[0]["content_cleaner_type"] == "FakeContentCleaner"


class TestContentValidationLogic:
    """Test content validation logic in _process_batch."""

    def test_article_with_sufficient_content_after_cleaning(self, monkeypatch):
        """Test that articles with >=150 chars non-boilerplate are saved."""
        rows = [
            (
                "cand-1",
                "https://example.com/article",
                "example.com",
                "article",
                "Example Site",
            )
        ]

        class FakeSession:
            def __init__(self):
                self.insert_calls = []
                self.update_calls = []
                self.commit_calls = 0

            def execute(self, query, params=None):
                # Track INSERT INTO articles
                if hasattr(query, "text") and "INSERT INTO articles" in str(query):
                    self.insert_calls.append(params)
                # Track UPDATE candidate_links
                elif hasattr(query, "text") and "UPDATE candidate_links" in str(query):
                    self.update_calls.append(params)
                elif params and "limit_with_buffer" in params:
                    return Mock(fetchall=lambda: rows)
                return Mock(fetchall=lambda: [], scalar=lambda: None)

            def commit(self):
                self.commit_calls += 1

            def close(self):
                pass

            def expire_all(self):
                pass

            def rollback(self):
                pass

        class FakeDBManager:
            def __init__(self):
                self.session = FakeSession()

        class FakeExtractor:
            def _check_rate_limit(self, domain):
                return False

            def extract_content(self, *args, **kwargs):
                # Return content with 200 chars (will have 150+ after cleaning)
                return {
                    "title": "Test Article",
                    "content": "A" * 200,  # 200 chars of real content
                    "author": "Test Author",
                    "metadata": {},
                }

            def get_driver_stats(self):
                return {"has_persistent_driver": False}

        class FakeBylineCleaner:
            def clean_byline(self, *args, **kwargs):
                return {"authors": ["Test Author"], "wire_services": []}

        class FakeContentCleaner:
            def process_single_article(self, text, domain, dry_run=False):
                # Simulate removing 30 chars of boilerplate, leaving 170 chars
                cleaned_text = text[:170] if len(text) > 170 else text
                return cleaned_text, {}

        class FakeTelemetry:
            def record_extraction(self, *args, **kwargs):
                pass

        class FakeMetrics:
            def __init__(self, *args, **kwargs):
                pass

            def set_content_type_detection(self, *args):
                pass

            def finalize(self, *args):
                pass

        monkeypatch.setattr(extraction, "DatabaseManager", FakeDBManager)
        monkeypatch.setattr(extraction, "BylineCleaner", FakeBylineCleaner)
        monkeypatch.setattr(extraction, "ExtractionMetrics", FakeMetrics)
        monkeypatch.setattr(extraction, "calculate_content_hash", lambda *a: "hash123")
        monkeypatch.setattr(
            extraction,
            "ContentTypeDetector",
            lambda **kw: Mock(detect=lambda **k: None),
        )

        db = FakeDBManager()
        args = Namespace(dump_sql=False)
        extractor = FakeExtractor()
        byline_cleaner = FakeBylineCleaner()
        content_cleaner = FakeContentCleaner()
        telemetry = FakeTelemetry()

        result = extraction._process_batch(
            args,
            extractor,
            byline_cleaner,
            content_cleaner,
            telemetry,
            per_batch=1,
            batch_num=1,
            host_403_tracker={},
            domains_for_cleaning=defaultdict(list),
            db=db,
        )

        # Verify article was inserted (sufficient content after cleaning)
        assert len(db.session.insert_calls) == 1
        assert result["processed"] == 1

    def test_article_with_insufficient_content_after_cleaning_skipped(
        self, monkeypatch
    ):
        """Test that articles with <150 chars non-boilerplate are skipped."""
        rows = [
            (
                "cand-1",
                "https://stltoday.com/article",
                "stltoday.com",
                "article",
                "STL Today",
            )
        ]

        class FakeSession:
            def __init__(self):
                self.insert_calls = []
                self.update_calls = []
                self.commit_calls = 0

            def execute(self, query, params=None):
                if hasattr(query, "text") and "INSERT INTO articles" in str(query):
                    self.insert_calls.append(params)
                elif hasattr(query, "text") and "UPDATE candidate_links" in str(query):
                    self.update_calls.append(params)
                elif params and "limit_with_buffer" in params:
                    return Mock(fetchall=lambda: rows)
                return Mock(fetchall=lambda: [], scalar=lambda: None)

            def commit(self):
                self.commit_calls += 1

            def close(self):
                pass

            def expire_all(self):
                pass

            def rollback(self):
                pass

        class FakeDBManager:
            def __init__(self):
                self.session = FakeSession()

        class FakeExtractor:
            def _check_rate_limit(self, domain):
                return False

            def extract_content(self, *args, **kwargs):
                # Return mostly boilerplate content (185 chars total)
                return {
                    "title": "DANCING WITH THE STARS",
                    "content": (
                        "Get up-to-the-minute news sent straight to your device.\n"
                        "CAPTCHA\n"
                        "Subscribe to continue reading\n"
                        "Log in to your account\n"
                        "Already a subscriber?\n"
                    ),
                    "author": "Eric Mccandless",
                    "metadata": {},
                }

            def get_driver_stats(self):
                return {"has_persistent_driver": False}

        class FakeBylineCleaner:
            def clean_byline(self, *args, **kwargs):
                return {"authors": ["Eric Mccandless"], "wire_services": []}

        class FakeContentCleaner:
            def process_single_article(self, text, domain, dry_run=False):
                # Simulate aggressive boilerplate removal (only 50 chars remain)
                # This simulates the persistent_boilerplate_patterns matching
                # stltoday.com login form patterns
                cleaned_text = "DANCING WITH THE STARS\nEric Mccandless"  # 42 chars
                # Return metadata indicating subscription patterns were found
                return cleaned_text, {
                    "patterns_matched": ["subscription", "paywall"],
                    "persistent_removals": 3,
                }

        class FakeTelemetry:
            def record_extraction(self, *args, **kwargs):
                pass

        class FakeMetrics:
            def __init__(self, *args, **kwargs):
                pass

            def set_content_type_detection(self, *args):
                pass

            def finalize(self, *args):
                pass

        monkeypatch.setattr(extraction, "DatabaseManager", FakeDBManager)
        monkeypatch.setattr(extraction, "BylineCleaner", FakeBylineCleaner)
        monkeypatch.setattr(extraction, "ExtractionMetrics", FakeMetrics)
        monkeypatch.setattr(extraction, "calculate_content_hash", lambda *a: "hash123")
        monkeypatch.setattr(
            extraction,
            "ContentTypeDetector",
            lambda **kw: Mock(detect=lambda **k: None),
        )

        db = FakeDBManager()
        args = Namespace(dump_sql=False)
        extractor = FakeExtractor()
        byline_cleaner = FakeBylineCleaner()
        content_cleaner = FakeContentCleaner()
        telemetry = FakeTelemetry()

        result = extraction._process_batch(
            args,
            extractor,
            byline_cleaner,
            content_cleaner,
            telemetry,
            per_batch=1,
            batch_num=1,
            host_403_tracker={},
            domains_for_cleaning=defaultdict(list),
            db=db,
        )

        # Verify article WAS inserted with status='paywall' (new behavior)
        assert len(db.session.insert_calls) == 1
        inserted_article = db.session.insert_calls[0]
        assert inserted_article["status"] == "paywall"
        # Verify candidate_link was still marked as 'extracted' to avoid retry
        assert len(db.session.update_calls) >= 1
        assert result["processed"] == 1

    def test_content_validation_with_empty_content(self, monkeypatch):
        """Test content validation handles empty/None content gracefully."""
        rows = [
            (
                "cand-1",
                "https://example.com/article",
                "example.com",
                "article",
                "Example Site",
            )
        ]

        class FakeSession:
            def __init__(self):
                self.insert_calls = []
                self.update_calls = []

            def execute(self, query, params=None):
                if hasattr(query, "text") and "INSERT INTO articles" in str(query):
                    self.insert_calls.append(params)
                elif hasattr(query, "text") and "UPDATE candidate_links" in str(query):
                    self.update_calls.append(params)
                elif params and "limit_with_buffer" in params:
                    return Mock(fetchall=lambda: rows)
                return Mock(fetchall=lambda: [], scalar=lambda: None)

            def commit(self):
                pass

            def close(self):
                pass

            def expire_all(self):
                pass

            def rollback(self):
                pass

        class FakeDBManager:
            def __init__(self):
                self.session = FakeSession()

        class FakeExtractor:
            def _check_rate_limit(self, domain):
                return False

            def extract_content(self, *args, **kwargs):
                # Return None for content (extraction failure)
                return {
                    "title": "Test Article",
                    "content": None,
                    "author": None,
                    "metadata": {},
                }

            def get_driver_stats(self):
                return {"has_persistent_driver": False}

        class FakeBylineCleaner:
            def clean_byline(self, *args, **kwargs):
                return {"authors": [], "wire_services": []}

        class FakeContentCleaner:
            def process_single_article(self, text, domain, dry_run=False):
                # Should not be called with empty content
                return "", {}

        class FakeTelemetry:
            def record_extraction(self, *args, **kwargs):
                pass

        class FakeMetrics:
            def __init__(self, *args, **kwargs):
                pass

            def set_content_type_detection(self, *args):
                pass

            def finalize(self, *args):
                pass

        monkeypatch.setattr(extraction, "DatabaseManager", FakeDBManager)
        monkeypatch.setattr(extraction, "BylineCleaner", FakeBylineCleaner)
        monkeypatch.setattr(extraction, "ExtractionMetrics", FakeMetrics)
        monkeypatch.setattr(extraction, "calculate_content_hash", lambda *a: "hash123")
        monkeypatch.setattr(
            extraction,
            "ContentTypeDetector",
            lambda **kw: Mock(detect=lambda **k: None),
        )

        db = FakeDBManager()
        args = Namespace(dump_sql=False)
        extractor = FakeExtractor()
        byline_cleaner = FakeBylineCleaner()
        content_cleaner = FakeContentCleaner()
        telemetry = FakeTelemetry()

        extraction._process_batch(
            args,
            extractor,
            byline_cleaner,
            content_cleaner,
            telemetry,
            per_batch=1,
            batch_num=1,
            host_403_tracker={},
            domains_for_cleaning=defaultdict(list),
            db=db,
        )

        # Verify NO article was inserted (empty content)
        assert len(db.session.insert_calls) == 0
        # Verify candidate marked as extracted (don't retry empty content)
        assert len(db.session.update_calls) >= 1


@pytest.mark.postgres
@pytest.mark.integration
class TestContentValidationWithPersistentPatterns:
    """Test content validation with actual persistent_boilerplate_patterns table."""

    def test_persistent_patterns_strip_boilerplate_before_validation(
        self, cloud_sql_session
    ):
        """Test that persistent boilerplate patterns are used for content validation.

        This integration test verifies:
        1. BalancedBoundaryContentCleaner queries persistent_boilerplate_patterns
        2. Domain-specific patterns are applied during content validation
        3. Articles with insufficient non-boilerplate content are skipped
        """
        from sqlalchemy import text

        from src.models import CandidateLink, Source
        from src.utils.content_cleaner_balanced import BalancedBoundaryContentCleaner

        # Create test source
        source = Source(
            id="source-test-boilerplate",
            host="testboilerplate.com",
            host_norm="testboilerplate.com",
            canonical_name="Test Boilerplate News",
            status="active",
        )
        cloud_sql_session.add(source)

        # Create candidate link
        candidate = CandidateLink(
            id="cand-test-boilerplate",
            url="https://testboilerplate.com/article",
            source="testboilerplate.com",
            source_id=source.id,
            status="article",
        )
        cloud_sql_session.add(candidate)
        cloud_sql_session.commit()

        # Insert persistent boilerplate pattern for this domain
        cloud_sql_session.execute(
            text(
                """
                INSERT INTO persistent_boilerplate_patterns (
                    id, domain, pattern_type, pattern_text, pattern_text_hash,
                    occurrence_count, is_active
                ) VALUES (
                    :id, :domain, :pattern_type, :pattern_text, :pattern_text_hash,
                    :occurrences, :is_active
                )
                """
            ),
            {
                "id": "pattern-login-form",
                "domain": "testboilerplate.com",
                "pattern_type": "subscription",
                "pattern_text": "Get up-to-the-minute news sent straight to your device",
                "pattern_text_hash": 123456789,
                "occurrences": 10,
                "is_active": True,
            },
        )
        cloud_sql_session.commit()

        # Test content with mostly boilerplate
        content_text = (
            "Get up-to-the-minute news sent straight to your device\n"
            "Subscribe to continue reading\n"
            "This is actual article content but it's too short"
        )

        # Use BalancedBoundaryContentCleaner to strip boilerplate
        # Mock telemetry to return our persistent pattern
        from unittest.mock import Mock

        mock_telemetry = Mock()
        mock_telemetry.get_persistent_patterns.return_value = [
            {
                "text_content": "Get up-to-the-minute news sent straight to your device",
                "pattern_type": "subscription",
                "confidence_score": 0.95,
                "occurrences_total": 10,
                "removal_reason": "Persistent subscription pattern",
            }
        ]

        cleaner = BalancedBoundaryContentCleaner(enable_telemetry=True)
        cleaner.telemetry = mock_telemetry
        stripped_content, metadata = cleaner.process_single_article(
            text=content_text,
            domain="testboilerplate.com",
            dry_run=True,
        )

        # Verify boilerplate was removed
        assert "Get up-to-the-minute news" not in stripped_content
        # Verify remaining content is less than MIN_CONTENT_LENGTH (150)
        assert len(stripped_content.strip()) < 150

        # Verify pattern lookup was called
        mock_telemetry.get_persistent_patterns.assert_called_once_with(
            "testboilerplate.com"
        )
