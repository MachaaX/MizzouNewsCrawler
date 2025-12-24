"""Tests for extract-url command parser."""

from src.cli.commands.extract_url import add_extract_url_parser


def test_add_extract_url_parser_creates_subcommand():
    """Verify extract-url subparser is created with correct arguments."""
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Add the extract-url subparser
    add_extract_url_parser(subparsers)

    # Parse valid arguments
    args = parser.parse_args(["extract-url", "https://example.com/article"])

    assert args.url == "https://example.com/article"
    assert hasattr(args, "func")
