import logging
import os
from unittest.mock import Mock, patch

from src.crawler import ContentExtractor, UNBLOCK_MIN_HTML_BYTES


def test_unblock_proxy_does_not_log_password(caplog, monkeypatch):
    # Ensure unblock env vars are set with a password
    monkeypatch.setenv("UNBLOCK_PROXY_URL", "https://unblock.decodo.com:60000")
    monkeypatch.setenv("UNBLOCK_PROXY_USER", "testuser")
    monkeypatch.setenv("UNBLOCK_PROXY_PASS", "testpass")
    monkeypatch.setenv("UNBLOCK_PREFER_API_POST", "true")

    extractor = ContentExtractor()
    # Simulate POST-first flow returning a large HTML (success)
    large_resp = Mock()
    large_resp.status_code = 200
    large_resp.text = "<html>" + ("x" * UNBLOCK_MIN_HTML_BYTES) + "</html>"

    caplog.set_level(logging.INFO)
    with patch("requests.post", return_value=large_resp):
        # Run extraction; this should use POST and set proxy metadata
        extractor._extract_with_unblock_proxy("https://example.com/article", None, None)

    # Ensure logs do not contain raw unblock password
    assert "testpass" not in caplog.text
    assert "UNBLOCK_PROXY_PASS" not in caplog.text
