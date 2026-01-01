"""Tests for Squid-only proxy configuration.

Ensures every proxy code path now routes exclusively through the Squid proxy.
"""

import os
from unittest.mock import Mock, patch

import pytest

from src.crawler import ContentExtractor
from src.crawler.proxy_config import ProxyProvider


class TestSquidOnlyProxySystem:
    """Test suite for Squid-only proxy routing."""

    @pytest.fixture
    def squid_env(self, monkeypatch):
        """Set up Squid proxy environment."""
        monkeypatch.setenv("SQUID_PROXY_URL", "http://t9880447.eero.online:3128")
        monkeypatch.setenv("PROXY_PROVIDER", "direct")  # Should be overridden to squid

    def test_squid_provider_override(self, squid_env):
        """Test that provider is correctly overridden to SQUID when Squid URL is set."""
        extractor = ContentExtractor()

        assert extractor.proxy_manager.active_provider == ProxyProvider.SQUID
        assert extractor.session.proxies["http"] == "http://t9880447.eero.online:3128"
        assert extractor.session.proxies["https"] == "http://t9880447.eero.online:3128"

    def test_squid_without_env_var(self, monkeypatch):
        """Test behavior when SQUID_PROXY_URL is not set."""
        # Don't set SQUID_PROXY_URL
        monkeypatch.setenv("PROXY_PROVIDER", "direct")

        extractor = ContentExtractor()

        # Should still route proxy traffic to Squid (default URL)
        assert "t9880447.eero.online:3128" in str(extractor.session.proxies)

    @patch("requests.get")
    def test_unblock_proxy_uses_squid(self, mock_get, squid_env):
        """Test that unblock proxy method always uses Squid."""
        # Mock successful Squid response (must be > 1000 chars)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = (
            "<html><body><h1>Test Article</h1>"
            + "<p>Content here</p>" * 100
            + "</body></html>"
        )  # > 1000 chars
        mock_get.return_value = mock_response

        extractor = ContentExtractor()
        result = extractor._extract_with_unblock_proxy("https://example.com/article")

        # Verify request was made through Squid proxy
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["proxies"]["http"] == "http://t9880447.eero.online:3128"
        assert call_kwargs["proxies"]["https"] == "http://t9880447.eero.online:3128"

        # Verify result indicates Squid method
        assert result["method"] == "squid_proxy"

    def test_domain_sessions_use_squid(self, squid_env):
        """Test that domain-specific sessions also route through Squid."""
        extractor = ContentExtractor()

        # Get a domain session (this triggers proxy setup)
        session = extractor._get_domain_session("https://example.com")

        # Verify domain session also uses Squid proxy
        assert session.proxies["http"] == "http://t9880447.eero.online:3128"
        assert session.proxies["https"] == "http://t9880447.eero.online:3128"

    def test_no_decodo_code_paths_active(self, squid_env):
        """Test that no legacy Decodo code paths are triggered."""
        with patch("src.crawler.proxy_config.os.getenv") as mock_getenv:
            # Mock environment to simulate legacy Decodo credentials present
            def side_effect(key, default=None):
                if key == "SQUID_PROXY_URL":
                    return "http://t9880447.eero.online:3128"
                elif key in [
                    "DECODO_USERNAME",
                    "DECODO_PASSWORD",
                    "UNBLOCK_PROXY_USER",
                    "UNBLOCK_PROXY_PASS",
                ]:
                    return "dummy_value"  # Legacy creds present but should be ignored
                return default

            mock_getenv.side_effect = side_effect

            extractor = ContentExtractor()

            # Despite legacy creds being present, should still use SQUID provider
            assert extractor.proxy_manager.active_provider == ProxyProvider.SQUID

    @patch("requests.get")
    def test_all_proxy_methods_use_squid(self, mock_get, squid_env):
        """Integration test: All proxy-enabled extraction methods should use Squid."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = (
            "<html><body><h1>Title</h1>" + "<p>Content</p>" * 100 + "</body></html>"
        )  # > 1000 chars
        mock_get.return_value = mock_response

        extractor = ContentExtractor()

        # Test different extraction scenarios that would use proxies

        # 1. Unblock proxy extraction
        extractor._extract_with_unblock_proxy("https://example.com/unblock")

        # 2. Regular HTTP extraction with proxy
        # This would go through the session proxy setup
        extractor._get_domain_session("https://example.com")

        # Verify all use Squid
        assert all(
            "t9880447.eero.online:3128" in str(call) for call in mock_get.call_args_list
        )

    def test_squid_provider_enum_exists(self):
        """Test that SQUID provider type is properly defined."""
        assert hasattr(ProxyProvider, "SQUID")
        assert ProxyProvider.SQUID.value == "squid"

    @patch("requests.get")
    def test_squid_proxy_error_handling(self, mock_get, squid_env):
        """Test error handling when Squid proxy fails."""
        from src.crawler import ProxyChallengeError

        # Mock Squid returning small response (blocked)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Blocked"  # Too small (< 1000 chars)
        mock_get.return_value = mock_response

        extractor = ContentExtractor()

        with pytest.raises(ProxyChallengeError):
            extractor._extract_with_unblock_proxy("https://example.com/blocked")

    @patch("requests.get")
    def test_squid_proxy_challenge_detection(self, mock_get, squid_env):
        """Test Squid proxy detects challenge pages properly."""
        from src.crawler import ProxyChallengeError

        mock_response = Mock()
        mock_response.status_code = 200
        # Make response long enough to pass size check but still trigger challenge detection
        mock_response.text = (
            "Access to this page has been denied by security policy." + " " * 1000
        )
        mock_get.return_value = mock_response

        extractor = ContentExtractor()

        with pytest.raises(ProxyChallengeError) as exc_info:
            extractor._extract_with_unblock_proxy("https://example.com/challenge")

        # Should detect challenge page pattern
        assert "challenge_page" in str(exc_info.value)

    def test_backward_compatibility_env_vars(self, monkeypatch):
        """Test that old proxy environment variables are properly overridden."""
        # Set old-style proxy vars
        monkeypatch.setenv("PROXY_PROVIDER", "decodo")
        monkeypatch.setenv("DECODO_USERNAME", "old_user")
        monkeypatch.setenv("UNBLOCK_PROXY_URL", "https://old.decodo.com:60000")

        # Set Squid override
        monkeypatch.setenv("SQUID_PROXY_URL", "http://t9880447.eero.online:3128")

        extractor = ContentExtractor()

        # Should ignore all old vars and use Squid
        assert extractor.proxy_manager.active_provider == ProxyProvider.SQUID
        assert "t9880447.eero.online:3128" in str(extractor.session.proxies)
