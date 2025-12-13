"""CLI command module for extracting a single URL.

This module is a thin wrapper around the extraction logic implemented in
`src.cli.commands.extraction` that allows extracting a single URL from the
command line as a top-level command.
"""
from __future__ import annotations

import argparse

from .extraction import handle_extract_url_command


def add_extract_url_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "extract-url", help="Extract content for a single URL and save to DB"
    )
    parser.add_argument("url", type=str, help="URL to extract")
    parser.add_argument(
        "--source", type=str, default=None, help="Optional source name (publisher)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset slug to attach to candidate link",
    )
    parser.add_argument(
        "--dump-sql",
        dest="dump_sql",
        action="store_true",
        default=False,
        help="Dump SQL statements and parameters before executing (diagnostic)",
    )
    parser.add_argument(
        "--verify-insert",
        dest="verify_insert",
        action="store_true",
        default=False,
        help=(
            "After committing an inserted article, run a SELECT to verify the "
            "row exists and log a mismatch (diagnostic)."
        ),
    )
    parser.set_defaults(func=handle_extract_url_command)
    return parser
