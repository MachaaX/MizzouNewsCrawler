"""Runtime compatibility shims for third-party dependencies."""

from __future__ import annotations

import os
import tempfile


def _ensure_newspaper_cache_directory() -> None:
    """Provide the legacy ``CACHE_DIRECTORY`` constant expected by consumers."""

    try:
        import importlib

        settings = importlib.import_module("newspaper.settings")
    except Exception:  # pragma: no cover - optional dependency
        return

    if hasattr(settings, "CACHE_DIRECTORY"):
        return

    cache_dir: str | os.PathLike[str] | None = getattr(settings, "ANCHOR_DIRECTORY", None)
    if not cache_dir:
        top_dir = getattr(settings, "TOP_DIRECTORY", None)
        if top_dir:
            cache_name = getattr(settings, "CF_CACHE_DIRECTORY", "feed_category_cache")
            cache_dir = os.path.join(top_dir, cache_name)
        else:
            cache_dir = os.path.join(tempfile.gettempdir(), "newspaper_cache")

    path_str = os.fspath(cache_dir)
    setattr(settings, "CACHE_DIRECTORY", path_str)

    try:
        os.makedirs(path_str, exist_ok=True)
    except Exception:
        # Directory creation failures are non-fatal; downstream code will raise if needed.
        pass


try:  # pragma: no cover - executed on interpreter startup
    _ensure_newspaper_cache_directory()
except Exception:
    # sitecustomize must never block interpreter startup; swallow unexpected errors.
    pass
