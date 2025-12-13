"""Utility helpers for proxy URL handling and masking.

These helpers are designed to provide safe, consistent formatting for
proxy URLs when logging or writing them to telemetry. They intentionally
avoid exposing credentials in logs.
"""

from __future__ import annotations

from urllib.parse import urlparse


def mask_proxy_url(proxy: str | None) -> str | None:
    """Return a proxy URL with any password redacted.

    Examples:
        https://user:pass@unblock.decodo.com:60000 -> https://user:***@unblock.decodo.com:60000
        https://unblock.decodo.com:60000 -> https://unblock.decodo.com:60000

    Returns None if proxy is None or empty.
    """
    if not proxy:
        return None

    try:
        parsed = urlparse(proxy)
        if parsed.username:
            hostname = parsed.hostname or parsed.netloc
            port = f":{parsed.port}" if parsed.port else ""
            scheme = f"{parsed.scheme}://" if parsed.scheme else ""
            return f"{scheme}{parsed.username}:***@{hostname}{port}"
        # No username -> safe to return as-is
        return proxy
    except Exception:
        # Best-effort safety: avoid returning something that may contain creds
        return "<redacted>"
