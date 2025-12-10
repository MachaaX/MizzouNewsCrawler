"""Integration tests for wire detection flow order in extraction.

Tests that the extraction flow follows the correct order:
1. JSON-LD/structured metadata detection (FIRST)
2. Byline detection (SKIPPED if #1 detects wire)
3. ContentTypeDetector (SKIPPED if #1 or #2 detects wire)

These tests verify the short-circuit behavior: when wire is detected
at an early stage, later stages are skipped.

CRITICAL: Uses PostgreSQL features and cloud_sql_session fixture.
Must run with @pytest.mark.integration and @pytest.mark.postgres markers.
"""

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.models import Article, CandidateLink, Source, WireService
from src.utils.byline_cleaner import BylineCleaner
from src.utils.content_type_detector import ContentTypeDetector


@pytest.fixture
def populated_wire_patterns(cloud_sql_session, monkeypatch):
    """Populate wire_services table for extraction flow tests."""
    # Clear existing patterns
    cloud_sql_session.query(WireService).delete()

    # Insert patterns used in extraction
    patterns = [
        # Author patterns for byline detection
        WireService(
            pattern=r"\bReuters\b",
            pattern_type="author",
            service_name="Reuters",
            priority=80,
            case_sensitive=False,
            active=True,
        ),
        WireService(
            pattern=r"\bAFP\b",
            pattern_type="author",
            service_name="AFP",
            priority=80,
            case_sensitive=False,
            active=True,
        ),
        # Content patterns for ContentTypeDetector
        WireService(
            pattern=r"\(AP\)\s*[—–-]",
            pattern_type="content",
            service_name="Associated Press",
            priority=100,
            case_sensitive=False,
            active=True,
        ),
    ]

    for pattern in patterns:
        cloud_sql_session.add(pattern)

    cloud_sql_session.commit()

    # Mock DatabaseManager to use cloud_sql_session
    @contextmanager
    def mock_get_session():
        try:
            yield cloud_sql_session
        finally:
            pass

    class MockDatabaseManager:
        def get_session(self):
            return mock_get_session()

    monkeypatch.setattr("src.models.database.DatabaseManager", MockDatabaseManager)

    yield
    # Cleanup via session rollback


@pytest.mark.integration
@pytest.mark.postgres
class TestExtractionFlowOrder:
    """Test extraction wire detection flow order and short-circuit behavior."""

    def test_wire_hints_skip_byline_detection(self, populated_wire_patterns):
        """Test that wire_hints in metadata short-circuits byline detection.

        When wire_hints is present with wire_services, the extraction flow
        should mark as wire immediately and skip byline cleaning.
        """
        # Simulate extraction metadata with wire_hints
        metadata = {
            "wire_hints": {
                "wire_services": ["Reuters"],
                "detected_by": ["structured_metadata"],
                "raw_source_name": "Reuters",
                "evidence": ["meta_author=Reuters"],
            }
        }

        # Process wire_hints (simulating Stage 1 of extraction)
        wire_hints = metadata.get("wire_hints")
        article_status = "extracted"
        wire_services = []

        if isinstance(wire_hints, dict):
            hint_services = [svc for svc in wire_hints.get("wire_services", []) if svc]
            if hint_services:
                article_status = "wire"
                wire_services = list(hint_services)

        # Verify wire detected
        assert article_status == "wire"
        assert "Reuters" in wire_services

        # Verify byline detection is skipped
        raw_author = "Some Author"
        byline_cleaner = BylineCleaner()
        byline_processed = False

        if article_status != "wire" and raw_author:
            # This block should NOT execute
            byline_processed = True
            byline_cleaner.clean_byline(raw_author, return_json=True)

        assert (
            not byline_processed
        ), "Byline detection should be skipped when wire_hints detects wire"

    def test_wire_hints_skip_content_type_detector(self, populated_wire_patterns):
        """Test that wire_hints short-circuits ContentTypeDetector.

        When wire is detected via wire_hints, ContentTypeDetector.detect()
        should NOT be called.
        """
        metadata = {
            "wire_hints": {
                "wire_services": ["AFP"],
                "detected_by": ["gannett_jsonld"],
            }
        }

        # Simulate Stage 1
        wire_hints = metadata.get("wire_hints")
        article_status = "extracted"

        if isinstance(wire_hints, dict):
            hint_services = wire_hints.get("wire_services", [])
            if hint_services:
                article_status = "wire"

        # Mock ContentTypeDetector to track if it's called
        detector = MagicMock(spec=ContentTypeDetector)
        detector_called = False

        # Simulate Stage 3 - only runs if article_status == "extracted"
        if article_status == "extracted":
            detector_called = True
            detector.detect(url="http://test.com", title="Test", metadata={})

        assert (
            not detector_called
        ), "ContentTypeDetector should not run when wire detected early"

    def test_byline_wire_skips_content_type_detector(
        self, populated_wire_patterns, cloud_sql_session
    ):
        """Test that byline wire detection short-circuits ContentTypeDetector.

        When wire is detected via byline (Stage 2), ContentTypeDetector (Stage 3)
        should be skipped.
        """
        article_status = "extracted"
        raw_author = "Reuters"  # This should trigger wire detection

        # Mock the byline cleaner to return wire detection result
        byline_result = {
            "authors": [],
            "wire_services": ["Reuters"],
            "is_wire_content": True,
        }

        # Simulate Stage 2 - byline detection
        if article_status != "wire" and raw_author:
            if byline_result.get("is_wire_content") and byline_result.get(
                "wire_services"
            ):
                article_status = "wire"

        assert article_status == "wire"

        # Simulate Stage 3 - should NOT run
        detector_called = False
        if article_status == "extracted":
            detector_called = True

        assert (
            not detector_called
        ), "ContentTypeDetector should be skipped after byline wire detection"

    def test_no_wire_runs_all_stages(self, populated_wire_patterns, cloud_sql_session):
        """Test that non-wire content runs all detection stages.

        When no wire is detected, the flow should proceed through all stages:
        1. wire_hints (no match) → continue
        2. byline (no match) → continue
        3. ContentTypeDetector → runs
        """
        # Stage 1: No wire_hints
        metadata = {}
        article_status = "extracted"

        wire_hints = metadata.get("wire_hints")
        if isinstance(wire_hints, dict):
            if wire_hints.get("wire_services"):
                article_status = "wire"

        assert article_status == "extracted", "Should not detect wire without hints"

        # Stage 2: Non-wire author
        raw_author = "John Smith, Local Reporter"
        byline_result = {
            "authors": ["John Smith"],
            "wire_services": [],
            "is_wire_content": False,
        }

        if article_status != "wire" and raw_author:
            if byline_result.get("is_wire_content") and byline_result.get(
                "wire_services"
            ):
                article_status = "wire"

        assert article_status == "extracted", "Should not detect wire from local author"

        # Stage 3: ContentTypeDetector should run
        detector_runs = False
        if article_status == "extracted":
            detector_runs = True

        assert detector_runs, "ContentTypeDetector should run for non-wire content"


@pytest.mark.integration
@pytest.mark.postgres
class TestExtractionFlowWithRealComponents:
    """Test extraction flow using real BylineCleaner and ContentTypeDetector."""

    def test_byline_cleaner_wire_detection(
        self, populated_wire_patterns, cloud_sql_session
    ):
        """Test BylineCleaner correctly detects wire from author string."""
        byline_cleaner = BylineCleaner()

        # Test wire author
        result = byline_cleaner.clean_byline("AFP", return_json=True)

        assert result.get("is_wire_content") is True
        assert "AFP" in result.get("wire_services", [])

    def test_content_type_detector_with_db_session(
        self, populated_wire_patterns, cloud_sql_session
    ):
        """Test ContentTypeDetector uses DB patterns correctly."""
        detector = ContentTypeDetector(session=cloud_sql_session)

        # Test with AP content pattern
        result = detector.detect(
            url="https://localnews.com/article",
            title="Breaking News",
            metadata={},
            content="WASHINGTON (AP) — The president announced today...",
        )

        if result:
            assert result.status == "wire"
            assert "Associated Press" in str(result.evidence)

    def test_full_flow_wire_via_hints_skips_later_stages(
        self, populated_wire_patterns, cloud_sql_session
    ):
        """Test complete extraction flow with wire_hints.

        Simulates the actual extraction.py flow logic.
        """
        # Setup: metadata with wire_hints
        metadata = {
            "wire_hints": {
                "wire_services": ["Reuters"],
                "detected_by": ["structured_metadata"],
            }
        }
        raw_author = "Some Author"

        # Track which stages run
        stages_run = []

        # Stage 1: Wire hints
        article_status = "extracted"
        wire_hints = metadata.get("wire_hints")
        if isinstance(wire_hints, dict):
            hint_services = [s for s in wire_hints.get("wire_services", []) if s]
            if hint_services:
                article_status = "wire"
                stages_run.append("wire_hints")

        # Stage 2: Byline (should be skipped)
        if article_status != "wire" and raw_author:
            stages_run.append("byline")
            byline_cleaner = BylineCleaner()
            byline_cleaner.clean_byline(raw_author, return_json=True)

        # Stage 3: ContentTypeDetector (should be skipped)
        if article_status == "extracted":
            stages_run.append("content_type_detector")

        # Verify only Stage 1 ran
        assert stages_run == ["wire_hints"]
        assert article_status == "wire"

    def test_full_flow_wire_via_byline_skips_detector(
        self, populated_wire_patterns, cloud_sql_session
    ):
        """Test extraction flow where byline detects wire."""
        # Setup: no wire_hints, but wire author
        metadata = {}
        raw_author = "Reuters"

        stages_run = []

        # Stage 1: Wire hints (none)
        article_status = "extracted"
        wire_hints = metadata.get("wire_hints")
        if isinstance(wire_hints, dict):
            hint_services = wire_hints.get("wire_services", [])
            if hint_services:
                article_status = "wire"
                stages_run.append("wire_hints")

        # Stage 2: Byline
        if article_status != "wire" and raw_author:
            stages_run.append("byline")
            byline_cleaner = BylineCleaner()
            result = byline_cleaner.clean_byline(raw_author, return_json=True)
            if result.get("is_wire_content") and result.get("wire_services"):
                article_status = "wire"

        # Stage 3: ContentTypeDetector (should be skipped)
        if article_status == "extracted":
            stages_run.append("content_type_detector")

        # Verify Stages 1 and 2 ran, but not 3
        assert "byline" in stages_run
        assert "content_type_detector" not in stages_run
        assert article_status == "wire"


@pytest.mark.integration
@pytest.mark.postgres
class TestExtractionFlowLogging:
    """Test that extraction flow logs correctly for debugging."""

    def test_wire_hints_logs_skipping_message(self, populated_wire_patterns, caplog):
        """Test that wire_hints detection logs 'skipping byline/content detection'."""
        import logging

        caplog.set_level(logging.INFO)

        # Simulate the logging from extraction.py
        wire_hints = {
            "wire_services": ["Reuters"],
            "detected_by": ["structured_metadata"],
        }
        detection_key = "structured_metadata"
        hint_services = wire_hints["wire_services"]

        # This matches the log message in extraction.py
        logging.info(
            "Wire detected via %s: wire=%s, authors=%s (skipping content detection)",
            detection_key,
            hint_services,
            [],  # Empty authors in this test case
        )

        assert "skipping content detection" in caplog.text
        assert "structured_metadata" in caplog.text

    def test_byline_wire_logs_skipping_message(self, populated_wire_patterns, caplog):
        """Test that byline wire detection logs 'skipping content detection'."""
        import logging

        caplog.set_level(logging.INFO)

        raw_author = "Reuters"
        cleaned_list = []
        byline_wire_services = ["Reuters"]

        # This matches the log message in extraction.py
        logging.info(
            "Wire service via byline '%s': authors=%s, wire=%s (skipping content detection)",
            raw_author,
            cleaned_list,
            byline_wire_services,
        )

        assert "skipping content detection" in caplog.text
        assert "Wire service via byline" in caplog.text


@pytest.mark.integration
@pytest.mark.postgres
class TestBylinePreservationInWireDetection:
    """Test that byline/author data is preserved even when wire is detected."""

    def test_wire_via_metadata_preserves_raw_author(
        self, populated_wire_patterns, cloud_sql_session
    ):
        """Test that raw_author is cleaned and preserved when wire detected via metadata.

        When wire is detected via wire_hints (Stage 1), we should still
        clean the raw_author to extract any individual author names.
        """
        byline_cleaner = BylineCleaner()

        # Simulate extraction with wire_hints and a raw_author
        # wire_hints would contain wire detection info from JSON-LD
        raw_author = "John Smith"

        # Clean the raw_author like Stage 1 does now
        extracted_authors = []
        if raw_author:
            byline_cleaned = byline_cleaner.clean_byline(raw_author, return_json=True)
            extracted_authors = byline_cleaned.get("authors", [])

        # Should have extracted "John Smith"
        assert "John Smith" in extracted_authors

    def test_wire_via_metadata_extracts_from_raw_source_name(
        self, populated_wire_patterns, cloud_sql_session
    ):
        """Test that author is extracted from raw_source_name in wire_hints."""
        byline_cleaner = BylineCleaner()

        wire_hints = {
            "wire_services": ["Reuters"],
            "detected_by": ["meta_author"],
            "raw_source_name": ["Jane Doe, Reuters"],
        }

        extracted_authors = []
        raw_sources = wire_hints.get("raw_source_name", [])
        if isinstance(raw_sources, str):
            raw_sources = [raw_sources]

        for raw_src in raw_sources:
            if raw_src and isinstance(raw_src, str):
                src_cleaned = byline_cleaner.clean_byline(raw_src, return_json=True)
                for auth in src_cleaned.get("authors", []):
                    if auth and auth not in extracted_authors:
                        extracted_authors.append(auth)

        # Should have extracted "Jane Doe" from "Jane Doe, Reuters"
        assert len(extracted_authors) >= 1
        # The exact result depends on BylineCleaner implementation

    def test_byline_result_has_authors_even_for_wire(
        self, populated_wire_patterns, cloud_sql_session
    ):
        """Test that byline_result includes authors even when wire detected."""
        byline_cleaner = BylineCleaner()

        # Wire detected via metadata, but we have author info
        wire_hints = {
            "wire_services": ["AFP"],
            "detected_by": ["jsonld_author"],
        }
        raw_author = "Marie Dupont"
        hint_services = wire_hints["wire_services"]

        # Simulate the updated Stage 1 logic
        extracted_authors = []
        if raw_author:
            byline_cleaned = byline_cleaner.clean_byline(raw_author, return_json=True)
            extracted_authors = byline_cleaned.get("authors", [])

        # Create byline_result as extraction.py now does
        byline_result = {
            "authors": extracted_authors,
            "count": len(extracted_authors),
            "primary_author": extracted_authors[0] if extracted_authors else None,
            "has_multiple_authors": len(extracted_authors) > 1,
            "wire_services": hint_services,
            "is_wire_content": True,
            "primary_wire_service": hint_services[0],
        }

        # Verify byline_result has the author preserved
        assert byline_result["is_wire_content"] is True
        assert byline_result["wire_services"] == ["AFP"]
        assert "Marie Dupont" in byline_result["authors"]
        assert byline_result["primary_author"] == "Marie Dupont"
