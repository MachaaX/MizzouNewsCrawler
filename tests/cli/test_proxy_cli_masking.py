import io
import sys
from unittest import mock

from src.cli.commands.proxy import handle_proxy_list
from src.crawler import proxy_config as proxy_config_module


def test_cli_proxy_list_masks_urls(monkeypatch):
    # Set up a standard proxy URL with credentials
    monkeypatch.setenv("STANDARD_PROXY_URL", "http://user:pass@standard.proxy:8080")
    monkeypatch.setenv("PROXY_PROVIDER", "standard")

    # Reset global ProxyManager singleton to pick up our env var changes
    proxy_config_module._proxy_manager = None

    # Capture stdout
    buf = io.StringIO()
    with mock.patch("sys.stdout", buf):
        handle_proxy_list(None)

    output = buf.getvalue()
    assert "http://user:***@standard.proxy:8080" in output
    # Ensure plaintext password not present
    assert "user:pass@" not in output
