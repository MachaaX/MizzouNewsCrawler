import re
from urllib.parse import urlparse, urlunparse

from storysniffer import StorySniffer


FILE_EXTENSION_MARKERS = (
    ".jpg",
    ".png",
    ".gif",
    ".pdf",
    ".css",
    ".js",
    ".xml",
    ".jpeg",
    ".svg",
    ".json",
)


def _normalize_url_for_patterns(raw_url: str | None) -> str:
    """Lowercase relevant URL components while normalizing directory paths."""

    if not raw_url:
        return ""

    candidate = raw_url.strip()
    if not candidate:
        return ""

    if "://" not in candidate:
        candidate = f"https://{candidate}"

    parsed = urlparse(candidate)

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = (parsed.path or "/").lower()
    params = parsed.params.lower()
    query = parsed.query.lower()
    fragment = parsed.fragment.lower()

    if path and not path.endswith("/"):
        last_segment = path.rsplit("/", 1)[-1]
        if not last_segment.endswith(FILE_EXTENSION_MARKERS):
            path = f"{path}/"

    normalized = urlunparse((scheme, netloc, path, params, query, fragment))
    return normalized


def check_is_article(url, discovery_method="unknown"):
    """Conservative article detection focusing on URL path structure patterns."""
    url_lower = _normalize_url_for_patterns(url)
    
    # Non-article URL patterns - designed to catch obvious non-article pages
    # Removed overly aggressive patterns like /category/, /tag/, /page/ to reduce false negatives
    # File extensions checked separately to ensure proper matching after normalization
    non_article_patterns = [
        "/search/",
        "/author/", 
        "/rss/",
        "/feed/",
        "/sitemap/",
        "/contact/",
        "/about/",
        "/privacy/",
        "/advertise/",
        "/advert/",
        ".jpg",
        ".png",
        ".gif",
        ".pdf",
        ".css",
        ".js",
        ".xml",
        ".jpeg",
        ".svg",
        ".json"
    ]
    for pattern in non_article_patterns:
        if pattern in url_lower:
            return False

    # Additional multimedia content filtering - these are typically not news articles
    # Using consistent pattern matching: some with path segments (/video/), some without (/watch)
    if "/video/" in url_lower or "/watch/" in url_lower or "/videos/" in url_lower:
        return False
    
    # Filter audio/podcast content
    if '/audio/' in url_lower or '/listen/' in url_lower or '/podcast/' in url_lower or '/podcasts/' in url_lower:
        return False
    
    # Article-like patterns
    if re.search(r"/stories?/[^/]+", url_lower):
        return True

    date_patterns = [r"/\d{4}/\d{1,2}/\d{1,2}/", r"/\d{4}-\d{1,2}-\d{1,2}/"]
    for pattern in date_patterns:
        if re.search(pattern, url_lower):
            return True

    article_section_patterns = [
        r"/news/[^/]+",
        r"/articles?/[^/]+",
        r"/content/[^/]+",
        r"/posts?/[^/]+",
        r"/blog/[^/]+",
    ]
    for pattern in article_section_patterns:
        if re.search(pattern, url_lower):
            return True

    if re.search(r"/\d{3,}", url_lower):
        return True

    # Newspaper4k special-case logic removed: empirically, URLs discovered via newspaper4k
    # are not different enough from other discovery methods to justify separate handling.
    # Previous attempts at special-casing newspaper4k URLs increased false positives/negatives.

    # Final fallback: use StorySniffer for additional article detection
    # Note: Using .guess() method - the .is_article_url() method does not exist in StorySniffer API
    try:
        sniffer = StorySniffer()
        return bool(sniffer.guess(url))
    except Exception:
        return False
