"""Tests for cleanup-candidates CLI command."""

import argparse
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.cli.commands.cleanup_candidates import (
    add_cleanup_candidates_parser,
    handle_cleanup_candidates_command,
)
from src.models import CandidateLink


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



