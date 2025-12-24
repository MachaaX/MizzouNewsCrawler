"""Tests for discovery-status CLI command."""

import argparse

import pytest

from src.cli.commands.discovery_status import (
    _to_int,
    add_discovery_status_parser,
)


class TestToInt:
    """Tests for _to_int helper function."""

    def test_to_int_with_integer(self):
        """Converts integer to int."""
        assert _to_int(42) == 42

    def test_to_int_with_string(self):
        """Converts string to int."""
        assert _to_int("123") == 123

    def test_to_int_with_float_string(self):
        """Float string returns default since int() raises ValueError."""
        assert _to_int("45.7") == 0

    def test_to_int_with_none(self):
        """Returns default for None."""
        assert _to_int(None) == 0
        assert _to_int(None, default=10) == 10

    def test_to_int_with_invalid_string(self):
        """Returns default for invalid string."""
        assert _to_int("invalid") == 0
        assert _to_int("abc", default=5) == 5


class TestAddDiscoveryStatusParser:
    """Tests for add_discovery_status_parser."""

    def test_creates_parser_with_correct_name(self):
        """Parser is created with discovery-status name."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        
        result = add_discovery_status_parser(subparsers)
        
        assert result is not None
        args = parser.parse_args(["discovery-status"])
        assert hasattr(args, "dataset")
        assert hasattr(args, "verbose")
        assert hasattr(args, "func")

    def test_dataset_argument_optional(self):
        """Dataset argument is optional."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_discovery_status_parser(subparsers)
        
        args = parser.parse_args(["discovery-status"])
        assert args.dataset is None

    def test_accepts_dataset_argument(self):
        """Accepts dataset filter argument."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_discovery_status_parser(subparsers)
        
        args = parser.parse_args(["discovery-status", "--dataset", "mizzou"])
        assert args.dataset == "mizzou"

    def test_verbose_flag_default_false(self):
        """Verbose flag defaults to False."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_discovery_status_parser(subparsers)
        
        args = parser.parse_args(["discovery-status"])
        assert args.verbose is False

    def test_verbose_flag_can_be_set(self):
        """Verbose flag can be enabled."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_discovery_status_parser(subparsers)
        
        args = parser.parse_args(["discovery-status", "-v"])
        assert args.verbose is True

    def test_verbose_long_form(self):
        """Verbose flag accepts long form."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_discovery_status_parser(subparsers)
        
        args = parser.parse_args(["discovery-status", "--verbose"])
        assert args.verbose is True
