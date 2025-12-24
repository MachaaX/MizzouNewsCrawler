"""
Tests for content cleaner session reuse functionality.

Verifies that the content cleaner properly reuses existing database sessions
to prevent transaction conflicts and connection pool exhaustion.
"""

import pytest
from unittest.mock import MagicMock, Mock, patch, call
from sqlalchemy import text as sql_text

from src.utils.content_cleaner_balanced import BalancedBoundaryContentCleaner
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
                result = cleaner.analyze_domain("example.com")
                
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
            articles = cleaner._get_articles_for_domain("example.com")
            
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
            patterns = cleaner._get_persistent_patterns_for_domain("example.com")
            
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
