"""Unit tests for CMS metadata extraction from JavaScript data objects.

Tests the _extract_cms_metadata_from_html method that captures title,
author, and other fields from CMS-specific JavaScript objects like
NXSTdata.content (Nexstar), dataLayer (Gray TV), and JSON-LD.
"""

import pytest

from src.crawler import ContentExtractor


@pytest.fixture
def extractor():
    """Create ContentExtractor instance for testing."""
    return ContentExtractor(timeout=5)


class TestNexstarMetadataExtraction:
    """Test extraction from Nexstar NXSTdata.content patterns."""

    def test_extracts_title_and_author(self, extractor):
        """Test extraction of title and authorName from NXSTdata.content."""
        html = """
        <script>
            window.NXSTdata = window.NXSTdata || {};
            window.NXSTdata.content = Object.assign( window.NXSTdata.content, {
                "title":"Fire District warns public not to burn",
                "authorName":"John Smith",
                "description":"Local fire warning issued",
                "publicationDate":"2025-12-09T12:17:04-06:00",
                "primaryCategory":"Local News"
            } )
        </script>
        """
        extractor._latest_cms_metadata = None
        extractor._extract_cms_metadata_from_html(html)

        assert extractor._latest_cms_metadata is not None
        assert extractor._latest_cms_metadata.get("title") == "Fire District warns public not to burn"
        assert extractor._latest_cms_metadata.get("author") == "John Smith"
        assert extractor._latest_cms_metadata.get("publish_date") == "2025-12-09T12:17:04-06:00"
        assert extractor._latest_cms_metadata.get("category") == "Local News"
        assert extractor._latest_cms_metadata.get("cms_source") == "nexstar"

    def test_handles_unicode_in_title(self, extractor):
        """Test handling of unicode characters in title."""
        html = """
        <script>
            window.NXSTdata.content = Object.assign( window.NXSTdata.content, {
                "title":"Fire District warns public about burning\\u00a0",
                "authorName":"Jane Doe"
            } )
        </script>
        """
        extractor._latest_cms_metadata = None
        extractor._extract_cms_metadata_from_html(html)

        assert extractor._latest_cms_metadata is not None
        # Unicode \u00a0 is non-breaking space
        assert "Fire District warns public about burning" in extractor._latest_cms_metadata.get("title", "")

    def test_ignores_empty_fields(self, extractor):
        """Test that empty fields are not stored."""
        html = """
        <script>
            window.NXSTdata.content = Object.assign( window.NXSTdata.content, {
                "title":"",
                "authorName":"John Smith",
                "description":""
            } )
        </script>
        """
        extractor._latest_cms_metadata = None
        extractor._extract_cms_metadata_from_html(html)

        # Should have author but not empty title
        if extractor._latest_cms_metadata:
            assert not extractor._latest_cms_metadata.get("title")
            assert extractor._latest_cms_metadata.get("author") == "John Smith"


class TestJsonLdMetadataExtraction:
    """Test extraction from JSON-LD when CMS patterns don't match."""

    def test_extracts_from_jsonld_newsarticle(self, extractor):
        """Test fallback to JSON-LD NewsArticle schema."""
        html = """
        <script type="application/ld+json">
        {
            "@type": "NewsArticle",
            "headline": "Breaking News Story",
            "author": {"@type": "Person", "name": "Reporter Name"},
            "datePublished": "2025-12-09T10:00:00Z"
        }
        </script>
        """
        extractor._latest_cms_metadata = None
        extractor._extract_cms_metadata_from_html(html)

        assert extractor._latest_cms_metadata is not None
        assert extractor._latest_cms_metadata.get("title") == "Breaking News Story"
        assert extractor._latest_cms_metadata.get("author") == "Reporter Name"
        assert extractor._latest_cms_metadata.get("publish_date") == "2025-12-09T10:00:00Z"
        assert extractor._latest_cms_metadata.get("cms_source") == "json_ld"

    def test_extracts_author_from_array(self, extractor):
        """Test extraction when author is an array."""
        html = """
        <script type="application/ld+json">
        {
            "@type": "NewsArticle",
            "headline": "Multi-author Story",
            "author": [
                {"@type": "Person", "name": "First Author"},
                {"@type": "Person", "name": "Second Author"}
            ]
        }
        </script>
        """
        extractor._latest_cms_metadata = None
        extractor._extract_cms_metadata_from_html(html)

        assert extractor._latest_cms_metadata is not None
        assert extractor._latest_cms_metadata.get("author") == "First Author"

    def test_extracts_author_as_string(self, extractor):
        """Test extraction when author is a simple string."""
        html = """
        <script type="application/ld+json">
        {
            "@type": "NewsArticle",
            "headline": "Simple Author Story",
            "author": "Staff Reporter"
        }
        </script>
        """
        extractor._latest_cms_metadata = None
        extractor._extract_cms_metadata_from_html(html)

        assert extractor._latest_cms_metadata is not None
        assert extractor._latest_cms_metadata.get("author") == "Staff Reporter"


class TestCmsMetadataPriority:
    """Test that Nexstar patterns take priority over JSON-LD."""

    def test_nexstar_takes_priority(self, extractor):
        """Test that Nexstar data takes priority over JSON-LD."""
        html = """
        <script>
            window.NXSTdata.content = Object.assign( window.NXSTdata.content, {
                "title":"Nexstar Title",
                "authorName":"Nexstar Author"
            } )
        </script>
        <script type="application/ld+json">
        {
            "@type": "NewsArticle",
            "headline": "JSON-LD Title",
            "author": "JSON-LD Author"
        }
        </script>
        """
        extractor._latest_cms_metadata = None
        extractor._extract_cms_metadata_from_html(html)

        assert extractor._latest_cms_metadata is not None
        # Nexstar should take priority
        assert extractor._latest_cms_metadata.get("title") == "Nexstar Title"
        assert extractor._latest_cms_metadata.get("author") == "Nexstar Author"
        assert extractor._latest_cms_metadata.get("cms_source") == "nexstar"


class TestNoMetadataFound:
    """Test behavior when no CMS metadata is found."""

    def test_no_metadata_returns_none(self, extractor):
        """Test that missing metadata doesn't create empty dict."""
        html = """
        <html>
        <head><title>Simple Page</title></head>
        <body><p>Content</p></body>
        </html>
        """
        extractor._latest_cms_metadata = None
        extractor._extract_cms_metadata_from_html(html)

        # Should remain None when no CMS data found
        assert extractor._latest_cms_metadata is None

    def test_malformed_json_handled(self, extractor):
        """Test that malformed JSON doesn't cause errors."""
        html = """
        <script>
            window.NXSTdata.content = Object.assign( window.NXSTdata.content, {
                "title": "Test",
                "broken": 
            } )
        </script>
        """
        extractor._latest_cms_metadata = None
        # Should not raise exception
        extractor._extract_cms_metadata_from_html(html)
