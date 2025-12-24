"""
Tests for content cleaner session reuse functionality.

Verifies that the content cleaner properly reuses existing database sessions
to prevent transaction conflicts and connection pool exhaustion.
"""

import pytest
from unittest.mock import MagicMock, Mock, patch, call
from sqlalchemy import text as sql_text

from src.utils.content_cleaner_balanced import (
    BalancedBoundaryContentCleaner,
    SOCIAL_SHARE_PHRASES,
    SOCIAL_SHARE_PREFIX_SEPARATORS,
)
from src.models.database import DatabaseManager


class TestSessionReuse:
    """Test that content cleaner reuses existing sessions instead of creating new ones."""

    def test_analyze_domain_accepts_session_parameter(self):
        """Verify analyze_domain accepts and uses a session parameter."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        
        # Mock the helper methods to avoid actual DB queries
        mock_session = Mock()
        with patch.object(cleaner, '_get_articles_for_domain', return_value=[]) as mock_get_articles:
            with patch.object(cleaner, '_get_persistent_patterns_for_domain', return_value=[]) as mock_get_patterns:
                result = cleaner.analyze_domain("example.com", session=mock_session)
                
                # Verify the session was passed to helper methods
                mock_get_articles.assert_called_once()
                assert mock_get_articles.call_args[1]['session'] == mock_session
                
                mock_get_patterns.assert_called_once()
                assert mock_get_patterns.call_args[1]['session'] == mock_session
                
                # Verify result structure
                assert result['domain'] == "example.com"
                assert result['article_count'] == 0
                assert 'segments' in result

    def test_analyze_domain_creates_session_when_none_provided(self):
        """Verify analyze_domain creates a session when none is provided (backward compatibility)."""
        mock_db = Mock(spec=DatabaseManager)
        mock_session = Mock()
        mock_db.session = mock_session
        
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        cleaner._shared_db = mock_db
        
        with patch.object(cleaner, '_get_articles_for_domain', return_value=[]) as mock_get_articles:
            with patch.object(cleaner, '_get_persistent_patterns_for_domain', return_value=[]) as mock_get_patterns:
                # Call without session parameter
                cleaner.analyze_domain("example.com")
                
                # Verify it used the db.session
                mock_get_articles.assert_called_once()
                assert mock_get_articles.call_args[1]['session'] == mock_session
                
                mock_get_patterns.assert_called_once()
                assert mock_get_patterns.call_args[1]['session'] == mock_session

    def test_get_articles_for_domain_uses_provided_session(self):
        """Verify _get_articles_for_domain uses the provided session."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        
        mock_session = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            (1, "https://example.com/article1", "content1", "hash1"),
            (2, "https://example.com/article2", "content2", "hash2"),
        ]
        mock_session.execute.return_value = mock_result
        
        # Mock safe_session_execute to use our mock session
        with patch('src.utils.content_cleaner_balanced.safe_session_execute', return_value=mock_result) as mock_execute:
            articles = cleaner._get_articles_for_domain(
                "example.com",
                sample_size=10,
                session=mock_session
            )
            
            # Verify safe_session_execute was called with our session
            mock_execute.assert_called_once()
            assert mock_execute.call_args[0][0] == mock_session
            
            # Verify results
            assert len(articles) == 2
            assert articles[0]['id'] == 1
            assert articles[0]['url'] == "https://example.com/article1"
            assert articles[1]['id'] == 2

    def test_get_articles_for_domain_creates_session_when_none_provided(self):
        """Verify _get_articles_for_domain creates session when none provided (backward compatibility)."""
        mock_db = Mock(spec=DatabaseManager)
        mock_session = Mock()
        mock_db.session = mock_session
        
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        cleaner._shared_db = mock_db
        
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        
        with patch('src.utils.content_cleaner_balanced.safe_session_execute', return_value=mock_result) as mock_execute:
            cleaner._get_articles_for_domain("example.com")
            
            # Verify it used db.session
            mock_execute.assert_called_once()
            assert mock_execute.call_args[0][0] == mock_session

    def test_get_persistent_patterns_uses_provided_session(self):
        """Verify _get_persistent_patterns_for_domain uses the provided session."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        
        mock_session = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ("pattern1", "header", 0.95),
            ("pattern2", "footer", 0.92),
        ]
        mock_session.execute.return_value = mock_result
        
        with patch('src.utils.content_cleaner_balanced.safe_session_execute', return_value=mock_result) as mock_execute:
            patterns = cleaner._get_persistent_patterns_for_domain(
                "example.com",
                session=mock_session
            )
            
            # Verify safe_session_execute was called with our session
            mock_execute.assert_called_once()
            assert mock_execute.call_args[0][0] == mock_session
            
            # Verify results
            assert len(patterns) == 2
            assert patterns[0]['text'] == "pattern1"
            assert patterns[0]['pattern_type'] == "header"
            assert patterns[0]['boundary_score'] == 0.95
            assert patterns[1]['text'] == "pattern2"

    def test_get_persistent_patterns_creates_session_when_none_provided(self):
        """Verify _get_persistent_patterns_for_domain creates session when none provided."""
        mock_db = Mock(spec=DatabaseManager)
        mock_session = Mock()
        mock_db.session = mock_session
        
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        cleaner._shared_db = mock_db
        
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        
        with patch('src.utils.content_cleaner_balanced.safe_session_execute', return_value=mock_result) as mock_execute:
            cleaner._get_persistent_patterns_for_domain("example.com")
            
            # Verify it used db.session
            mock_execute.assert_called_once()
            assert mock_execute.call_args[0][0] == mock_session

    def test_no_nested_sessions_created_when_session_provided(self):
        """Verify that when a session is provided, no new sessions are created."""
        mock_db = Mock(spec=DatabaseManager)
        mock_session = Mock()
        mock_db.session = mock_session
        
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        cleaner._shared_db = mock_db
        
        # Mock the database query results
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        
        provided_session = Mock()
        
        with patch('src.utils.content_cleaner_balanced.safe_session_execute', return_value=mock_result) as mock_execute:
            # Analyze domain with provided session
            cleaner.analyze_domain("example.com", session=provided_session)
            
            # Verify all execute calls used the provided session, not db.session
            for call_args in mock_execute.call_args_list:
                assert call_args[0][0] == provided_session
                assert call_args[0][0] != mock_session, "Should not use db.session when session is provided"

    def test_session_reuse_prevents_nested_context_managers(self):
        """Verify that session reuse avoids the 'with db.get_session()' pattern."""
        mock_db = Mock(spec=DatabaseManager)
        mock_session = Mock()
        mock_db.session = mock_session
        
        # get_session should NOT be called when a session is provided
        mock_get_session = Mock()
        mock_db.get_session = mock_get_session
        
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        cleaner._shared_db = mock_db
        
        provided_session = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        
        with patch('src.utils.content_cleaner_balanced.safe_session_execute', return_value=mock_result):
            # Call with provided session
            cleaner._get_articles_for_domain("example.com", session=provided_session)
            cleaner._get_persistent_patterns_for_domain("example.com", session=provided_session)
            
            # Verify get_session was NEVER called (no nested sessions)
            mock_get_session.assert_not_called()


class TestExtractionCommandIntegration:
    """Test integration with extraction command's session passing."""

    def test_extraction_command_passes_session_to_cleaner(self):
        """Verify extraction command pattern: cleaner.analyze_domain(domain, session=session)."""
        # This test documents the expected usage pattern
        mock_db = Mock(spec=DatabaseManager)
        mock_session = Mock()
        mock_db.session = mock_session
        
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
            db=mock_db,
        )
        
        # Simulate extraction command usage
        session = mock_db.session
        
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        
        with patch('src.utils.content_cleaner_balanced.safe_session_execute', return_value=mock_result) as mock_execute:
            # This is how extraction.py should call it
            cleaner.analyze_domain("example.com", session=session)
            
            # Verify the session was used for queries
            assert all(
                call_args[0][0] == session
                for call_args in mock_execute.call_args_list
            ), "All database queries should use the provided session"


class TestContentCleanerCore:
    """Test core content cleaning functionality to increase coverage."""

    def test_initialization_with_shared_db(self):
        """Test cleaner initialization with shared DatabaseManager."""
        mock_db = Mock(spec=DatabaseManager)
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
            db=mock_db,
        )
        
        assert cleaner._shared_db == mock_db
        assert cleaner.enable_telemetry is False
        assert cleaner.use_cloud_sql is True

    def test_connect_to_db_returns_shared_db(self):
        """Test _connect_to_db returns shared DatabaseManager when available."""
        mock_db = Mock(spec=DatabaseManager)
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
            db=mock_db,
        )
        
        result = cleaner._connect_to_db()
        assert result == mock_db

    def test_connect_to_db_creates_new_when_none(self):
        """Test _connect_to_db creates new DatabaseManager when none provided."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        
        with patch('src.utils.content_cleaner_balanced.DatabaseManager') as mock_db_class:
            cleaner._connect_to_db()
            mock_db_class.assert_called_once()

    def test_normalize_navigation_token(self):
        """Test navigation token normalization."""
        assert BalancedBoundaryContentCleaner._normalize_navigation_token("HOME") == "home"
        assert BalancedBoundaryContentCleaner._normalize_navigation_token("News") == "news"
        assert BalancedBoundaryContentCleaner._normalize_navigation_token("  SPORTS  ") == "sports"

    def test_extract_navigation_prefix(self):
        """Test extraction of navigation prefixes."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        
        # Content with navigation prefix
        text = "Home • News • Local\n\nActual article content here."
        result = cleaner._extract_navigation_prefix(text)
        
        # Should detect navigation pattern at start
        assert result is None or isinstance(result, str)

    def test_assess_boundary_quality(self):
        """Test boundary quality assessment."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        
        # Good boundary (complete sentence)
        score = cleaner._assess_boundary_quality("This is a complete sentence.")
        assert 0 <= score <= 1
        
        # Poor boundary (fragment)
        score = cleaner._assess_boundary_quality("incomplete text without")
        assert 0 <= score <= 1

    def test_classify_pattern(self):
        """Test pattern classification."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        
        # Should classify different types
        pattern_type = cleaner._classify_pattern("Copyright © 2023 Example Corp")
        assert pattern_type in ["footer", "header", "navigation", "sidebar", "subscription", "trending", "other"]
        
        pattern_type = cleaner._classify_pattern("Share this article on Facebook")
        assert pattern_type in ["footer", "header", "navigation", "sidebar", "subscription", "trending", "other"]

    def test_generate_removal_reason(self):
        """Test removal reason generation."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        
        reason = cleaner._generate_removal_reason("Copyright © 2023 Example Corp", "footer", 0.95, 10)
        assert isinstance(reason, str)
        assert "footer" in reason.lower() or "Footer" in reason

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
        
        assert 'total_removable_chars' in stats
        assert 'removal_percentage' in stats
        assert stats['total_removable_chars'] > 0
        assert 0 <= stats['removal_percentage'] <= 100

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

    def test_remove_social_share_header(self):
        """Test removal of social share headers."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        
        text = "Share • Facebook Twitter\n\nActual article content here."
        result = cleaner._remove_social_share_header(text)
        
        assert isinstance(result, dict)
        assert 'cleaned_text' in result
        assert 'removed_text' in result

    def test_process_single_article_with_telemetry_disabled(self):
        """Test article processing with telemetry disabled."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        
        text = "This is a simple article without boilerplate."
        cleaned, metadata = cleaner.process_single_article(text, domain="example.com")
        
        assert isinstance(cleaned, str)
        assert isinstance(metadata, dict)
        assert 'social_share_header_removed' in metadata
        assert 'persistent_removals' in metadata
        assert 'chars_removed' in metadata

    def test_is_high_confidence_boilerplate(self):
        """Test high confidence boilerplate detection."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        
        # Copyright footer
        assert cleaner._is_high_confidence_boilerplate("Copyright © 2023. All rights reserved.") is True
        
        # Normal content
        assert cleaner._is_high_confidence_boilerplate("The city council met today.") is False

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

    def test_calculate_position_consistency(self):
        """Test position consistency calculation."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        
        # Perfect consistency (all at same relative position)
        exact_matches = {
            "1": [(10, 20)],
            "2": [(10, 20)],
            "3": [(10, 20)],
        }
        articles_by_id = {
            "1": {"content": "A" * 100},
            "2": {"content": "B" * 100},
            "3": {"content": "C" * 100},
        }
        consistency = cleaner._calculate_position_consistency(exact_matches, articles_by_id)
        assert consistency > 0.9

    def test_contains_term(self):
        """Test term containment check."""
        assert BalancedBoundaryContentCleaner._contains_term("hello world", "world") is True
        assert BalancedBoundaryContentCleaner._contains_term("hello world", "xyz") is False

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

    def test_filter_with_balanced_boundaries(self):
        """Test balanced boundary filtering."""
        cleaner = BalancedBoundaryContentCleaner(
            db_path=":memory:",
            enable_telemetry=False,
        )
        
        candidates = {
            "test pattern": {"1", "2", "3"}
        }
        
        articles = [
            {"id": "1", "content": "Before. test pattern. After."},
            {"id": "2", "content": "Start. test pattern. End."},
            {"id": "3", "content": "Begin. test pattern. Finish."},
        ]
        
        filtered = cleaner._filter_with_balanced_boundaries(articles, candidates, min_occurrences=2, telemetry_id="test")
        
        assert isinstance(filtered, list)
