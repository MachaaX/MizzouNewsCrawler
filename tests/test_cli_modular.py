import argparse
import builtins
import sys
import types

import pytest

from src.cli import cli_modular


def _noop_setup_logging(_level: str) -> None:
    """Stub logging setup for CLI tests."""
    return None


def test_main_no_command_shows_help(capsys):
    exit_code = cli_modular.main([], setup_logging_func=_noop_setup_logging)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Available commands" in captured.err


def test_main_unknown_command(capsys):
    exit_code = cli_modular.main(
        ["unknown"],
        setup_logging_func=_noop_setup_logging,
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Unknown command" in captured.err


def test_main_uses_handler_override():
    def custom_handler(args):
        assert args.command == "custom"
        return 42

    exit_code = cli_modular.main(
        ["custom"],
        setup_logging_func=_noop_setup_logging,
        handler_overrides={"custom": custom_handler},
    )

    assert exit_code == 42


def test_main_loads_command_module_dynamically(monkeypatch):
    module_name = "src.cli.commands.verification"
    fake_module = types.ModuleType(module_name)

    def add_verify_urls_parser(subparsers):
        parser = subparsers.add_parser("verify-urls")
        parser.set_defaults(command="verify-urls")

    def handle_verification_command(args):
        assert args.command == "verify-urls"
        return 13

    fake_module.add_verify_urls_parser = add_verify_urls_parser
    fake_module.handle_verification_command = handle_verification_command

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == module_name:
            return fake_module
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setitem(sys.modules, module_name, fake_module)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    exit_code = cli_modular.main(
        ["verify-urls"],
        setup_logging_func=_noop_setup_logging,
    )

    assert exit_code == 13


def test_load_command_parser_fallback_functions(monkeypatch):
    module_name = "src.cli.commands.crawl"
    fake_module = types.ModuleType(module_name)

    def add_crawl_options_parser(subparsers):
        parser = subparsers.add_parser("crawl")
        parser.set_defaults(command="crawl")

    def handle_crawl_request_command(args):
        return 7

    fake_module.add_crawl_options_parser = add_crawl_options_parser
    fake_module.handle_crawl_request_command = handle_crawl_request_command

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == module_name:
            return fake_module
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    parser_func, handler_func = cli_modular._load_command_parser("crawl")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    parser_func(subparsers)
    args = argparse.Namespace(command="crawl")

    assert handler_func(args) == 7


def test_load_command_parser_module_not_found(monkeypatch):
    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        raise ModuleNotFoundError("missing module")

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert cli_modular._load_command_parser("telemetry") is None


def test_resolve_handler_prefers_existing_callable():
    sentinel = object()

    def handler(namespace):
        return sentinel

    resolved = cli_modular._resolve_handler(
        argparse.Namespace(func=handler, command="any")
    )

    assert resolved is handler
