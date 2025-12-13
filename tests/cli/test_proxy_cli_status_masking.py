import io
import sys
from unittest import mock

from src.cli.commands.proxy import handle_proxy_status
from src.crawler import proxy_config as proxy_config_module


def test_cli_proxy_status_masks_urls(monkeypatch):
    monkeypatch.setenv("STANDARD_PROXY_URL", "http://user:pass@standard.proxy:8080")
    monkeypatch.setenv("PROXY_PROVIDER", "standard")
    proxy_config_module._proxy_manager = None

    buf = io.StringIO()
    with mock.patch("sys.stdout", buf):
        handle_proxy_status(None)

    output = buf.getvalue()
    assert "http://user:***@standard.proxy:8080" in output
    assert "user:pass@" not in output
