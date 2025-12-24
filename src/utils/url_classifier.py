"""URL classification utilities for filtering non-article pages during discovery.

NOTE: URL pattern filtering has been migrated to the database (verification_patterns table).
The URLVerificationService loads patterns dynamically from the database with caching.
This module now serves as a lightweight fallback only.
"""

import re
from urllib.parse import urlparse

# DEPRECATED: Patterns have been migrated to verification_patterns table.
# This fallback list is kept for compatibility but should not be extended.
# To add new patterns, insert into verification_patterns table instead.
NON_ARTICLE_PATTERNS = []

# Compile patterns for efficiency
COMPILED_NON_ARTICLE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in NON_ARTICLE_PATTERNS
]


def is_likely_article_url(url: str) -> bool:
    """Check if a URL is likely to be an article page.

    Returns False for obvious non-article pages like galleries, categories,
    static pages, etc. Returns True otherwise (may still be a false positive,
    but filters out the most obvious non-article patterns).

    Args:
        url: The URL to classify

    Returns:
        True if the URL might be an article, False if it's clearly not

    Examples:
        >>> is_likely_article_url("https://example.com/news/story-title")
        True
        >>> is_likely_article_url("https://example.com/video-gallery/news")
        False
        >>> is_likely_article_url("https://example.com/category/sports")
        False
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.lower()

        # Check against non-article patterns
        for pattern in COMPILED_NON_ARTICLE_PATTERNS:
            if pattern.search(path):
                return False

        return True

    except Exception:
        # If parsing fails, be conservative and allow it
        return True


def classify_url_batch(urls: list[str]) -> tuple[list[str], list[str]]:
    """Classify a batch of URLs into likely articles and non-articles.

    Args:
        urls: List of URLs to classify

    Returns:
        Tuple of (likely_articles, filtered_out)
    """
    likely_articles = []
    filtered_out = []

    for url in urls:
        if is_likely_article_url(url):
            likely_articles.append(url)
        else:
            filtered_out.append(url)

    return likely_articles, filtered_out
