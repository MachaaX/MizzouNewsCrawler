"""Tests for wire service detection in content cleaner."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from src.utils.content_cleaner_balanced import BalancedBoundaryContentCleaner


class TestWireServiceDetection:
    """Tests for wire service detection functionality."""

    def test_detect_wire_service_in_pattern_no_match(self):
        """Test pattern without wire service."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )

        # Mock wire detector
        cleaner.wire_detector = Mock()
        cleaner.wire_detector._is_wire_service.return_value = False
        cleaner.wire_detector._detected_wire_services = []

        pattern_text = "Local newspaper footer content"
        result = cleaner._detect_wire_service_in_pattern(pattern_text, "example.com")

        assert result is None

    def test_is_high_confidence_boilerplate_copyright(self):
        """Test high confidence detection for copyright text."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )

        # Test copyright
        assert (
            cleaner._is_high_confidence_boilerplate(
                "Copyright © 2023. All rights reserved."
            )
            is True
        )

        # Test AP footer
        assert (
            cleaner._is_high_confidence_boilerplate(
                "Copyright 2023 The Associated Press. All rights reserved."
            )
            is True
        )

        # Test normal content
        assert (
            cleaner._is_high_confidence_boilerplate("The city council met today.")
            is False
        )

    def test_is_high_confidence_boilerplate_privacy_policy(self):
        """Test high confidence detection for privacy policy links."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )

        # Long privacy policy text is high confidence
        assert (
            cleaner._is_high_confidence_boilerplate(
                "Privacy Policy | Terms of Service | Contact Us"
            )
            is True
        )

    def test_extract_navigation_prefix_no_nav(self):
        """Test with non-navigation text."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )

        # Normal article text
        text = (
            "The local school board discussed budget issues at their monthly meeting."
        )
        result = cleaner._extract_navigation_prefix(text)

        # Should not detect navigation
        assert result is None or len(result) == 0


class TestHelperMethods:
    """Tests for helper methods in content cleaner."""

    def test_normalize_navigation_token(self):
        """Test navigation token normalization."""
        assert (
            BalancedBoundaryContentCleaner._normalize_navigation_token("HOME") == "home"
        )
        assert (
            BalancedBoundaryContentCleaner._normalize_navigation_token("Contact Us")
            == "contact"
        )
        assert (
            BalancedBoundaryContentCleaner._normalize_navigation_token("Subscribe!")
            == "subscribe"
        )

    def test_contains_term(self):
        """Test term containment check."""
        # Case insensitive containment
        assert (
            BalancedBoundaryContentCleaner._contains_term("hello world", "world")
            is True
        )
        assert (
            BalancedBoundaryContentCleaner._contains_term("hello world", "xyz") is False
        )

    def test_assess_boundary_quality_fragment(self):
        """Test boundary quality for sentence fragment."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )

        # Fragment without proper boundaries
        text = "incomplete thought without"
        score = cleaner._assess_boundary_quality(text)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_is_social_share_cluster(self):
        """Test detection of social share clusters."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )

        # Social share cluster
        assert cleaner._is_social_share_cluster("Facebook Twitter Share Email") is True

        # Not a social share cluster
        assert cleaner._is_social_share_cluster("Important local news today") is False

    def test_detect_social_share_prefix_end(self):
        """Test detection of social share prefix end."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )

        text = "Share • Facebook Twitter\n\nContent starts here"
        result = cleaner._detect_social_share_prefix_end(text)

        # Should return position or None
        assert result is None or isinstance(result, int)

    def test_remove_social_share_header(self):
        """Test removal of social share headers."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )

        text = "Share • Facebook Twitter\n\nActual article content here."
        result = cleaner._remove_social_share_header(text)

        assert isinstance(result, dict)
        assert "cleaned_text" in result
        assert "removed_text" in result

    def test_calculate_domain_stats(self):
        """Test domain statistics calculation."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )

        articles = [
            {"id": "1", "content": "A" * 1000},
            {"id": "2", "content": "B" * 1000},
        ]

        segments = [
            {"text": "A" * 100, "length": 100},
            {"text": "B" * 50, "length": 50},
        ]

        stats = cleaner._calculate_domain_stats(articles, segments)

        assert "total_removable_chars" in stats
        assert "removal_percentage" in stats
        assert stats["total_removable_chars"] > 0
        assert 0 <= stats["removal_percentage"] <= 100

    def test_find_rough_candidates(self):
        """Test rough candidate detection."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )

        articles = [
            {"id": "1", "content": "First article with some content."},
            {"id": "2", "content": "Second article with different content."},
        ]

        candidates = cleaner._find_rough_candidates(articles)

        assert isinstance(candidates, dict)
