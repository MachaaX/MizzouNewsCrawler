"""Tests for cleanup-candidates CLI command."""

import argparse
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.cli.commands.cleanup_candidates import (
    add_cleanup_candidates_parser,
    handle_cleanup_candidates_command,
)


class TestAddCleanupCandidatesParser:
    """Tests for add_cleanup_candidates_parser."""

    def test_creates_parser_with_correct_name(self):
        """Parser is created with cleanup-candidates name."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        result = add_cleanup_candidates_parser(subparsers)

        assert result is not None
        args = parser.parse_args(["cleanup-candidates"])
        assert hasattr(args, "days")
        assert hasattr(args, "dry_run")

    def test_default_days_is_seven(self):
        """Default days threshold is 7."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_cleanup_candidates_parser(subparsers)

        args = parser.parse_args(["cleanup-candidates"])
        assert args.days == 7

    def test_accepts_custom_days(self):
        """Accepts custom days argument."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_cleanup_candidates_parser(subparsers)

        args = parser.parse_args(["cleanup-candidates", "--days", "14"])
        assert args.days == 14

    def test_dry_run_flag_default_false(self):
        """Dry run flag defaults to False."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_cleanup_candidates_parser(subparsers)

        args = parser.parse_args(["cleanup-candidates"])
        assert args.dry_run is False

    def test_dry_run_flag_can_be_set(self):
        """Dry run flag can be enabled."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_cleanup_candidates_parser(subparsers)

        args = parser.parse_args(["cleanup-candidates", "--dry-run"])
        assert args.dry_run is True


class TestHandleCleanupCandidatesCommand:
    """Tests for handle_cleanup_candidates_command."""

    @patch("src.cli.commands.cleanup_candidates.DatabaseManager")
    def test_handle_with_no_expired_candidates(self, mock_db_class):
        """Returns 0 when no expired candidates found."""
        # Mock the database to return no results
        mock_session = MagicMock()
        mock_session.execute.return_value.fetchall.return_value = []
        mock_db = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_db.get_session.return_value.__exit__.return_value = None
        mock_db_class.return_value = mock_db

        # Create args
        args = MagicMock()
        args.days = 7
        args.dry_run = False

        # Run command
        result = handle_cleanup_candidates_command(args)

        assert result == 0
        mock_session.execute.assert_called()

    @patch("src.cli.commands.cleanup_candidates.DatabaseManager")
    def test_handle_with_expired_candidates_dry_run(self, mock_db_class):
        """Dry run shows results without modifying database."""
        # Mock expired candidates
        mock_result = MagicMock()
        mock_result.id = 1
        mock_result.source = "example.com"
        mock_result.url = "https://example.com/old"
        mock_result.created_at = datetime.now(timezone.utc) - timedelta(days=10)
        mock_result.age_days = 10.0

        mock_session = MagicMock()
        # First call returns expired candidates, second for breakdown with 3 values
        mock_session.execute.return_value.fetchall.side_effect = [
            [mock_result],  # Initial expired query
            [
                ("example.com", 1, datetime.now(timezone.utc) - timedelta(days=10))
            ],  # Breakdown query: source, count, oldest
        ]

        mock_db = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_db.get_session.return_value.__exit__.return_value = None
        mock_db_class.return_value = mock_db

        args = MagicMock()
        args.days = 7
        args.dry_run = True

        result = handle_cleanup_candidates_command(args)

        assert result == 0
        # Should NOT call commit in dry run mode
        mock_session.commit.assert_not_called()

    @patch("src.cli.commands.cleanup_candidates.DatabaseManager")
    def test_handle_with_expired_candidates_updates_status(self, mock_db_class):
        """Updates candidates to paused status."""
        # Mock expired candidates
        mock_result = MagicMock()
        mock_result.id = 1
        mock_result.source = "example.com"
        mock_result.url = "https://example.com/old"
        mock_result.created_at = datetime.now(timezone.utc) - timedelta(days=10)
        mock_result.age_days = 10.0

        mock_session = MagicMock()
        mock_session.execute.return_value.fetchall.side_effect = [
            [mock_result],  # Initial expired query
            [
                ("example.com", 1, datetime.now(timezone.utc) - timedelta(days=10))
            ],  # Breakdown query: source, count, oldest
        ]
        mock_session.execute.return_value.rowcount = 1

        mock_db = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_db.get_session.return_value.__exit__.return_value = None
        mock_db_class.return_value = mock_db

        args = MagicMock()
        args.days = 7
        args.dry_run = False

        result = handle_cleanup_candidates_command(args)

        assert result == 0
        # Should call commit when not in dry run
        mock_session.commit.assert_called()
