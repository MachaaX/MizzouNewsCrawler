import re

from storysniffer import StorySniffer


def check_is_article(url, discovery_method="unknown"):
    """Conservative article detection focusing on URL path structure patterns."""
    url_lower = (url or "").lower()

    # if url doesnt end with '/', add '/' to normalize
    # This removes false matching of patterns like '/feed' matching '/feeding-poultry-blog'
    # so we now use /feed/ to check instead of /feed
    if not url_lower.endswith('/'):
        url_lower += '/'
    
    # added '/' in the right end to patterns to complement normalization
    # removed '/category/', '/tag/', '/page/' and '/sections/' from below non-article patterns to reduce false negatives and be less aggressive
    # added '.jpeg', '.svg', '.json'
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

    # commented out below code, cuz could cause false negatives
    # if "/category/" in url_lower or "/tag/" in url_lower or "/topics/" in url_lower:
    #     return False

    if "/video/" in url_lower or "/watch/" in url_lower or "/videos/" in url_lower:
        return False
    
    # added audio patterns
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

    # commented out below code, cuz newspaper4k discovered urls are no different than others
    # if discovery_method == "newspaper4k":
    #     path = url_lower.split("://")[-1].split("?")[0]
    #     segments = [
    #         seg for seg in ("/" + "/".join(path.split("/")[1:])).split("/") if seg
    #     ]
    #     if len(segments) >= 2 or any("-" in seg for seg in segments):
    #         return True
    #     return False

    # Final fallback: try storysniffer if available
    try:
        sniffer = StorySniffer()
        return bool(sniffer.guess(url)) # is_article_url method replaced with guess method
    except Exception:
        return False
