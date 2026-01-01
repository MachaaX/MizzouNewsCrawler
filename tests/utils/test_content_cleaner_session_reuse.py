"""
Tests for content cleaner basic functionality (session reuse tests removed).

Note: The session reuse functionality tests were removed because they
tested features that are not yet implemented in the content cleaner.
"""

from unittest.mock import Mock, patch

import pytest

from src.utils.content_cleaner_balanced import BalancedBoundaryContentCleaner


class TestContentCleanerBasics:
    """Basic tests for content cleaner without session parameter expectations."""

    def test_cleaner_initialization(self):
        """Test that the content cleaner initializes correctly."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )

        assert cleaner.db_path == ":memory:"
        assert cleaner.enable_telemetry is False
        assert cleaner.wire_detector is not None

    def test_analyze_domain_with_no_articles(self):
        """Test analyze_domain returns empty result when no articles found."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )

        # Mock database methods to return no articles or patterns
        with patch.object(cleaner, "_get_articles_for_domain", return_value=[]):
            with patch.object(
                cleaner, "_get_persistent_patterns_for_domain", return_value=[]
            ):
                result = cleaner.analyze_domain("example.com")

                assert result["domain"] == "example.com"
                assert result["article_count"] == 0
                assert result["segments"] == []

    def test_connect_to_db_with_shared_db(self):
        """Test that _connect_to_db uses shared DatabaseManager when provided."""
        mock_db = Mock()
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
            db=mock_db,
        )

        result = cleaner._connect_to_db()
        assert result is mock_db

    def test_connect_to_db_creates_new_when_no_shared(self):
        """Test that _connect_to_db creates new DatabaseManager when none provided."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )

        with patch(
            "src.utils.content_cleaner_balanced.DatabaseManager"
        ) as mock_db_class:
            mock_db_instance = Mock()
            mock_db_class.return_value = mock_db_instance

            result = cleaner._connect_to_db()

            mock_db_class.assert_called_once()
            assert result is mock_db_instance
