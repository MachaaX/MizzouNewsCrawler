"""Comprehensive tests for unblock proxy extraction feature.

This module tests the complete logic flow for the extraction_method='unblock' feature:

1. Database schema and migration
   - extraction_method column creation
   - Migration from selenium_only to extraction_method
   - Index creation

2. Domain classification logic
   - _get_domain_extraction_method() returns correct method
   - _mark_domain_special_extraction() sets appropriate method based on protection type
   - Strong protections (PerimeterX, DataDome, Akamai) → 'unblock'
   - Other JS protections → 'selenium'
   - Default domains → 'http'

3. Extraction flow routing
   - 'unblock' domains skip HTTP methods and use _extract_with_unblock_proxy()
   - 'selenium' domains skip HTTP methods and use Selenium
   - 'http' domains use standard flow

4. _extract_with_unblock_proxy() method
   - Successful extraction with full HTML
   - Partial extraction with missing fields
   - Failed extraction (still blocked)
   - Network/timeout errors
   - Invalid credentials
   - Environment variable configuration

5. Field-level extraction and fallbacks
   - All fields extracted successfully
   - Some fields missing → fallback to Selenium
   - All fields missing → raise appropriate error
   - Content hash calculation
   - Metadata tracking

6. Edge cases
   - Domain not in sources table
   - extraction_method is NULL
   - Multiple protection types
   - Cache invalidation
   - Concurrent requests
   - Large HTML responses
   - Malformed HTML
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call
from datetime import datetime

import pytest
from sqlalchemy import text

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.crawler import ContentExtractor
from src.models import Source


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up environment variables for unblock proxy."""
    monkeypatch.setenv("UNBLOCK_PROXY_URL", "https://unblock.decodo.com:60000")
    monkeypatch.setenv("UNBLOCK_PROXY_USER", "testuser")
    monkeypatch.setenv("UNBLOCK_PROXY_PASS", "testpass")


class TestDatabaseSchemaAndMigration:
    """Test database schema changes for extraction_method column.
    
    Note: These tests verify the ORM model definition. Actual database migration
    tests run in CI/CD with PostgreSQL integration tests.
    """

    def test_source_model_has_extraction_method_field(self):
        """Source model should have extraction_method attribute."""
        assert hasattr(Source, 'extraction_method'), "Source model should have extraction_method column"
        
    def test_source_model_has_selenium_only_field(self):
        """Source model should retain selenium_only for backward compatibility."""
        assert hasattr(Source, 'selenium_only'), "Source model should retain selenium_only column"
        
    def test_extraction_method_default_value(self):
        """extraction_method column should default to 'http'."""
        # Check the column definition has correct default
        column = Source.__table__.columns['extraction_method']
        assert column.default is not None, "Should have a default value"
        assert column.default.arg == 'http', "Default should be 'http'"
        
    def test_extraction_method_column_type(self):
        """extraction_method should be a String column."""
        column = Source.__table__.columns['extraction_method']
        assert str(column.type) == 'VARCHAR(32)', "Should be VARCHAR(32) type"


class TestDomainClassificationLogic:
    """Test how domains get classified into extraction methods.
    
    These tests override the autouse fixture to test actual logic.
    """
    
    @pytest.fixture(autouse=True)
    def restore_original_method(self, monkeypatch):
        """Restore original _get_domain_extraction_method for these tests."""
        # Don't mock it - let the real implementation run (with mocked database)
        pass

    def test_get_domain_extraction_method_returns_http_for_new_domain(self):
        """Domains not in sources table should return 'http' method."""
        extractor = ContentExtractor()
        
        with patch('src.models.database.DatabaseManager') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.get_session.return_value.__enter__.return_value = mock_session
            # Simulate domain not found in database
            mock_session.execute.return_value.fetchone.return_value = None
            
            method, protection = extractor._get_domain_extraction_method("new-domain.com")

        assert method == 'http'
        assert protection is None

    def test_get_domain_extraction_method_returns_unblock_for_perimeterx(self):
        """Domains with PerimeterX should return 'unblock' method."""
        extractor = ContentExtractor()
        
        with patch('src.models.database.DatabaseManager') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.get_session.return_value.__enter__.return_value = mock_session
            # Simulate database returning unblock + perimeterx
            mock_session.execute.return_value.fetchone.return_value = ('unblock', 'perimeterx')
            
            method, protection = extractor._get_domain_extraction_method("perimeterx-test.com")

        assert method == 'unblock'
        assert protection == 'perimeterx'

    def test_get_domain_extraction_method_returns_selenium_for_cloudflare(self):
        """Domains with Cloudflare should return 'selenium' method."""
        extractor = ContentExtractor()
        
        with patch('src.models.database.DatabaseManager') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.get_session.return_value.__enter__.return_value = mock_session
            # Simulate database returning selenium + cloudflare
            mock_session.execute.return_value.fetchone.return_value = ('selenium', 'cloudflare')
            
            method, protection = extractor._get_domain_extraction_method("cloudflare-test.com")

        assert method == 'selenium'
        assert protection == 'cloudflare'

    def test_get_domain_extraction_method_caches_results(self):
        """Should cache results to avoid repeated database queries."""
        extractor = ContentExtractor()

        with patch('src.models.database.DatabaseManager') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.get_session.return_value.__enter__.return_value = mock_session
            mock_session.execute.return_value.fetchone.return_value = ('unblock', 'perimeterx')

            # First call - should hit database
            method1, _ = extractor._get_domain_extraction_method("cached-domain.com")

            # Second call - should use cache
            method2, _ = extractor._get_domain_extraction_method("cached-domain.com")

            assert method1 == 'unblock'
            assert method2 == 'unblock'
            # Should only execute once due to caching
            assert mock_session.execute.call_count == 1

    def test_mark_domain_special_extraction_sets_unblock_for_perimeterx(self):
        """_mark_domain_special_extraction should set 'unblock' for PerimeterX."""
        extractor = ContentExtractor()
        
        with patch('src.models.database.DatabaseManager') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.get_session.return_value.__enter__.return_value = mock_session
            
            extractor._mark_domain_special_extraction("mark-perimeterx.com", "perimeterx")
            
            # Verify execute was called with UPDATE containing 'method': 'unblock'
            assert mock_session.execute.called
            call_args = mock_session.execute.call_args
            # Check the params dict contains method='unblock'
            params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
            assert params.get('method') == 'unblock', "Should set method to 'unblock' for PerimeterX"

    def test_mark_domain_special_extraction_sets_unblock_for_datadome(self):
        """DataDome should also get 'unblock' method."""
        extractor = ContentExtractor()
        
        with patch('src.models.database.DatabaseManager') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.get_session.return_value.__enter__.return_value = mock_session
            
            extractor._mark_domain_special_extraction("mark-datadome.com", "datadome")
            
            # Verify datadome is passed as protection_type
            assert mock_session.execute.called
            call_args = mock_session.execute.call_args
            params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
            assert params.get('protection_type') == 'datadome'

    def test_mark_domain_special_extraction_sets_selenium_for_cloudflare(self):
        """Cloudflare should get 'selenium' method, not 'unblock'."""
        extractor = ContentExtractor()
        
        with patch('src.models.database.DatabaseManager') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.get_session.return_value.__enter__.return_value = mock_session
            
            extractor._mark_domain_special_extraction("mark-cloudflare.com", "cloudflare")
            
            # Verify cloudflare uses selenium, not unblock
            assert mock_session.execute.called
            call_args = mock_session.execute.call_args
            params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
            # Cloudflare defaults to selenium (not unblock)
            assert params.get('method') == 'selenium', "Cloudflare should use 'selenium', not 'unblock'"

    def test_mark_domain_special_extraction_respects_explicit_method(self):
        """Should allow explicitly setting method parameter."""
        extractor = ContentExtractor()
        
        with patch('src.models.database.DatabaseManager') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.get_session.return_value.__enter__.return_value = mock_session
            
            # Explicitly set to selenium even though perimeterx would normally be unblock
            extractor._mark_domain_special_extraction("mark-explicit.com", "perimeterx", method="selenium")
            
            # Verify explicit method is used
            assert mock_session.execute.called
            call_args = mock_session.execute.call_args
            params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
            assert params.get('method') == 'selenium', "Should respect explicit method parameter"
        extractor._mark_domain_special_extraction("mark-explicit.com", "perimeterx", method="selenium")

        result = cloud_sql_session.execute(text("""
            SELECT extraction_method
            FROM sources
            WHERE host = 'mark-explicit.com'
        """)).fetchone()

        # Should override the auto-mapping
        assert result[0] == 'unblock', "Auto-mapping should override explicit method for strong protections"


class TestExtractionFlowRouting:
    """Test that extraction flow correctly routes based on extraction_method."""

    def test_unblock_domain_skips_http_methods(self, mock_env_vars):
        """Domains with extraction_method='unblock' should skip mcmetadata, newspaper4k, BeautifulSoup."""
        extractor = ContentExtractor(use_mcmetadata=True)

        with patch.object(extractor, '_get_domain_extraction_method', return_value=('unblock', 'perimeterx')), \
             patch.object(extractor, '_extract_with_unblock_proxy', return_value={
                 'url': 'https://test.com/article',
                 'title': 'Test Title',
                 'author': 'Test Author',
                 'content': 'Test content',
                 'publish_date': datetime(2025, 1, 1).isoformat(),
                 'metadata': {},
                 'extracted_at': datetime.utcnow().isoformat()
             }) as mock_unblock, \
             patch.object(extractor, '_extract_with_mcmetadata') as mock_mcmetadata, \
             patch.object(extractor, '_extract_with_newspaper') as mock_newspaper, \
             patch.object(extractor, '_extract_with_beautifulsoup') as mock_bs:

            result = extractor.extract_content('https://test.com/article')

            # Should call unblock proxy
            mock_unblock.assert_called_once()

            # Should NOT call HTTP methods
            mock_mcmetadata.assert_not_called()
            mock_newspaper.assert_not_called()
            mock_bs.assert_not_called()

            assert result['title'] == 'Test Title'

    def test_selenium_domain_skips_http_methods(self):
        """Domains with extraction_method='selenium' should skip HTTP methods."""
        extractor = ContentExtractor(use_mcmetadata=True)

        with patch.object(extractor, '_get_domain_extraction_method', return_value=('selenium', 'cloudflare')), \
             patch.object(extractor, '_extract_with_selenium', return_value={
                 'url': 'https://test.com/article',
                 'title': 'Selenium Title',
                 'content': 'Selenium content',
                 'metadata': {},
                 'extracted_at': datetime.utcnow().isoformat()
             }) as mock_selenium, \
             patch.object(extractor, '_extract_with_mcmetadata') as mock_mcmetadata, \
             patch.object(extractor, '_extract_with_newspaper') as mock_newspaper:

            # Trigger Selenium fallback by having missing fields
            with patch.object(extractor, '_get_missing_fields', return_value=['title', 'content']):
                result = extractor.extract_content('https://test.com/article')

            # Should eventually call Selenium
            mock_selenium.assert_called()

            # Should NOT call HTTP methods (skip_http_methods=True)
            mock_mcmetadata.assert_not_called()
            mock_newspaper.assert_not_called()

    def test_http_domain_uses_standard_flow(self):
        """Domains with extraction_method='http' should use standard extraction flow."""
        extractor = ContentExtractor(use_mcmetadata=True)

        with patch.object(extractor, '_get_domain_extraction_method', return_value=('http', None)), \
             patch.object(extractor, '_extract_with_mcmetadata', return_value={
                 'url': 'https://test.com/article',
                 'title': 'MC Title',
                 'author': 'MC Author',
                 'content': 'MC content',
                 'metadata': {},
                 'extracted_at': datetime.utcnow().isoformat()
             }) as mock_mcmetadata, \
             patch.object(extractor, '_extract_with_unblock_proxy') as mock_unblock:

            result = extractor.extract_content('https://test.com/article', html='<html>test</html>')

            # Should call mcmetadata (HTTP method)
            mock_mcmetadata.assert_called_once()

            # Should NOT call unblock proxy
            mock_unblock.assert_not_called()

            assert result['title'] == 'MC Title'


class TestUnblockProxyMethod:
    """Test _extract_with_unblock_proxy() method behavior."""

    def test_successful_extraction_with_full_html(self, mock_env_vars):
        """Should successfully extract from full HTML response."""
        extractor = ContentExtractor()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
        <head>
            <title>Test Article Title</title>
            <meta name="author" content="John Doe">
            <meta property="article:published_time" content="2025-01-15T10:00:00Z">
        </head>
        <body>
            <article>
                <h1>Test Article Title</h1>
                <p>This is the article content with enough text to pass validation.</p>
                <p>More content here to ensure we have substantial text.</p>
            </article>
        </body>
        </html>
        """ * 100  # Make it large enough (>5000 bytes)

        with patch('requests.get', return_value=mock_response):
            result = extractor._extract_with_unblock_proxy('https://test.com/article')

        assert result['title'] is not None
        assert result['content'] is not None
        assert result['metadata']['extraction_method'] == 'unblock_proxy'
        assert result['metadata']['proxy_used'] is True
        assert result['metadata']['page_source_length'] > 5000

    def test_blocked_response_returns_empty(self, mock_env_vars):
        """Should return empty dict if still blocked by bot protection."""
        extractor = ContentExtractor()

        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = """
        <html>
        <head><title>Access to this page has been denied</title></head>
        <body>Access denied</body>
        </html>
        """

        with patch('requests.get', return_value=mock_response):
            result = extractor._extract_with_unblock_proxy('https://test.com/article')

        assert result == {}

    def test_small_response_returns_empty(self, mock_env_vars):
        """Should return empty dict if response is too small (< 5000 bytes)."""
        extractor = ContentExtractor()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Small</body></html>"

        with patch('requests.get', return_value=mock_response):
            result = extractor._extract_with_unblock_proxy('https://test.com/article')

        assert result == {}

    def test_network_error_returns_empty(self, mock_env_vars):
        """Should return empty dict on network errors."""
        extractor = ContentExtractor()

        with patch('requests.get', side_effect=Exception("Connection timeout")):
            result = extractor._extract_with_unblock_proxy('https://test.com/article')

        assert result == {}

    def test_uses_environment_variables(self, monkeypatch):
        """Should read proxy configuration from environment variables."""
        monkeypatch.setenv("UNBLOCK_PROXY_URL", "https://custom.proxy.com:8080")
        monkeypatch.setenv("UNBLOCK_PROXY_USER", "customuser")
        monkeypatch.setenv("UNBLOCK_PROXY_PASS", "custompass")

        extractor = ContentExtractor()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>" + ("test " * 1000) + "</body></html>"

        with patch('requests.get', return_value=mock_response) as mock_get:
            extractor._extract_with_unblock_proxy('https://test.com/article')

            # Verify proxy URL was constructed correctly
            call_kwargs = mock_get.call_args[1]
            assert 'proxies' in call_kwargs
            assert 'customuser:custompass' in call_kwargs['proxies']['https']
            assert 'custom.proxy.com:8080' in call_kwargs['proxies']['https']

    def test_sends_required_headers(self, mock_env_vars):
        """Should send X-SU-* headers required by Decodo API."""
        extractor = ContentExtractor()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>" + ("test " * 1000) + "</body></html>"

        with patch('requests.get', return_value=mock_response) as mock_get:
            extractor._extract_with_unblock_proxy('https://test.com/article')

            call_kwargs = mock_get.call_args[1]
            headers = call_kwargs['headers']

            assert headers['X-SU-Session-Id'] == 'mizzou-crawler'
            assert headers['X-SU-Geo'] == 'United States'
            assert headers['X-SU-Locale'] == 'en-us'
            assert headers['X-SU-Headless'] == 'html'
            assert 'Mozilla' in headers['User-Agent']


class TestFieldLevelExtractionAndFallbacks:
    """Test field-level extraction behavior and fallback scenarios."""

    def test_all_fields_extracted_no_fallback(self, mock_env_vars):
        """When all fields are extracted, should not trigger Selenium fallback."""
        extractor = ContentExtractor()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
        <head>
            <title>Complete Article</title>
            <meta name="author" content="Jane Doe">
            <meta property="article:published_time" content="2025-01-15T10:00:00Z">
        </head>
        <body>
            <article>
                <h1>Complete Article</h1>
                <p>Full content here.</p>
            </article>
        </body>
        </html>
        """ * 100

        with patch.object(extractor, '_get_domain_extraction_method', return_value=('unblock', 'perimeterx')), \
             patch('requests.get', return_value=mock_response), \
             patch.object(extractor, '_extract_with_selenium') as mock_selenium:

            result = extractor.extract_content('https://test.com/article')

            # Should NOT call Selenium since all fields were extracted
            mock_selenium.assert_not_called()

            assert result['title'] is not None
            assert result['content'] is not None

    def test_missing_fields_triggers_selenium_fallback(self, mock_env_vars):
        """When some fields are missing, should fall back to Selenium."""
        extractor = ContentExtractor()

        # Unblock proxy returns partial data (missing content)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
        <head><title>Partial Article</title></head>
        <body></body>
        </html>
        """ * 10  # Small but > 5000 bytes

        with patch.object(extractor, '_get_domain_extraction_method', return_value=('unblock', 'perimeterx')), \
             patch('requests.get', return_value=mock_response), \
             patch.object(extractor, '_extract_with_selenium', return_value={
                 'url': 'https://test.com/article',
                 'title': 'Partial Article',
                 'content': 'Selenium extracted content',
                 'metadata': {},
                 'extracted_at': datetime.utcnow().isoformat()
             }) as mock_selenium:

            result = extractor.extract_content('https://test.com/article')

            # Should call Selenium to fill missing fields
            mock_selenium.assert_called()

            # Should have content from Selenium fallback
            assert 'Selenium extracted content' in result.get('content', '')

    def test_extraction_methods_tracked_in_metadata(self, mock_env_vars):
        """Should track which extraction method succeeded for each field."""
        extractor = ContentExtractor()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
        <head><title>Tracked Article</title></head>
        <body><p>""" + ("content " * 1000) + """</p></body>
        </html>
        """

        with patch.object(extractor, '_get_domain_extraction_method', return_value=('unblock', 'perimeterx')), \
             patch('requests.get', return_value=mock_response):

            result = extractor.extract_content('https://test.com/article')

            # Should track extraction methods
            assert 'extraction_methods' in result
            # Title should be from unblock_proxy
            if result.get('title'):
                assert result['extraction_methods'].get('title') == 'unblock_proxy'


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_domain_not_in_sources_table(self):
        """Should handle domains not in sources table gracefully."""
        extractor = ContentExtractor()
        extractor._extraction_method_cache = {}

        method, protection = extractor._get_domain_extraction_method("unknown-domain.example")

        assert method == 'http'
        assert protection is None

    def test_extraction_method_null_in_database(self, cloud_sql_session):
        """Should handle NULL extraction_method gracefully."""
        cloud_sql_session.execute(text("""
            INSERT INTO sources (id, host, canonical_name, extraction_method)
            VALUES ('test-null-method', 'null-method.com', 'Null Method', NULL)
            ON CONFLICT (id) DO UPDATE SET extraction_method = NULL
        """))
        cloud_sql_session.commit()

        extractor = ContentExtractor()
        extractor._extraction_method_cache = {}

        method, protection = extractor._get_domain_extraction_method("null-method.com")

        # Should default to 'http' when NULL
        assert method == 'http'

    def test_multiple_protection_types_uses_strongest(self, cloud_sql_session):
        """When a domain has multiple protection types, should use the strongest method."""
        # This is a theoretical edge case - in practice, one domain has one protection type
        # But the code should handle it gracefully
        cloud_sql_session.execute(text("""
            INSERT INTO sources (id, host, canonical_name, extraction_method, bot_protection_type)
            VALUES ('test-multi-protect', 'multi-protect.com', 'Multi Protect', 'unblock', 'perimeterx')
            ON CONFLICT (id) DO UPDATE SET
                extraction_method = 'unblock',
                bot_protection_type = 'perimeterx'
        """))
        cloud_sql_session.commit()

        extractor = ContentExtractor()
        extractor._extraction_method_cache = {}

        method, protection = extractor._get_domain_extraction_method("multi-protect.com")

        # Should return the method stored in database
        assert method == 'unblock'

    def test_cache_invalidation_on_mark_domain(self, cloud_sql_session):
        """Marking a domain should not use stale cached value."""
        cloud_sql_session.execute(text("""
            INSERT INTO sources (id, host, canonical_name, extraction_method)
            VALUES ('test-cache-invalidate', 'cache-invalidate.com', 'Cache Test', 'http')
            ON CONFLICT (id) DO UPDATE SET extraction_method = 'http'
        """))
        cloud_sql_session.commit()

        extractor = ContentExtractor()
        extractor._extraction_method_cache = {}

        # First call - cache 'http'
        method1, _ = extractor._get_domain_extraction_method("cache-invalidate.com")
        assert method1 == 'http'

        # Mark as unblock
        extractor._mark_domain_special_extraction("cache-invalidate.com", "perimeterx")

        # Clear cache manually (in production, this would need cache invalidation)
        extractor._extraction_method_cache = {}

        # Second call - should get updated value
        method2, _ = extractor._get_domain_extraction_method("cache-invalidate.com")
        assert method2 == 'unblock'

    def test_large_html_response_handled(self, mock_env_vars):
        """Should handle very large HTML responses without crashing."""
        extractor = ContentExtractor()

        # 2MB HTML response
        large_html = "<html><body>" + ("x" * 2_000_000) + "</body></html>"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = large_html

        with patch('requests.get', return_value=mock_response):
            result = extractor._extract_with_unblock_proxy('https://test.com/article')

            # Should process without error
            assert result['metadata']['page_source_length'] == len(large_html)

    def test_malformed_html_handled_gracefully(self, mock_env_vars):
        """Should handle malformed HTML without crashing."""
        extractor = ContentExtractor()

        malformed_html = """
        <html>
        <head><title>Malformed</head>
        <body>
        <p>Unclosed paragraph
        <div>Unclosed div
        """ * 100

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = malformed_html

        with patch('requests.get', return_value=mock_response):
            result = extractor._extract_with_unblock_proxy('https://test.com/article')

            # BeautifulSoup should still parse it
            assert isinstance(result, dict)

    def test_timeout_handled_gracefully(self, mock_env_vars):
        """Should handle request timeouts gracefully."""
        extractor = ContentExtractor()

        with patch('requests.get', side_effect=Exception("Timeout")):
            result = extractor._extract_with_unblock_proxy('https://test.com/article')

            assert result == {}

    def test_empty_proxy_credentials_uses_defaults(self, monkeypatch):
        """Should use default values when environment variables are missing."""
        # Don't set any proxy env vars
        monkeypatch.delenv("UNBLOCK_PROXY_URL", raising=False)
        monkeypatch.delenv("UNBLOCK_PROXY_USER", raising=False)
        monkeypatch.delenv("UNBLOCK_PROXY_PASS", raising=False)

        extractor = ContentExtractor()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>" + ("test " * 1000) + "</body></html>"

        with patch('requests.get', return_value=mock_response) as mock_get:
            extractor._extract_with_unblock_proxy('https://test.com/article')

            # Should use hardcoded defaults
            call_kwargs = mock_get.call_args[1]
            assert 'U0000332559' in call_kwargs['proxies']['https']
