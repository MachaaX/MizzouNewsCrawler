"""Runtime compatibility shims for third-party dependencies."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


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
    settings.CACHE_DIRECTORY = path_str

    try:
        os.makedirs(path_str, exist_ok=True)
    except Exception:
        # Directory creation failures are non-fatal; downstream code will raise if needed.
        pass


def _ensure_src_on_sys_path() -> None:
    """Expose vendored packages (for example ``src/mcmetadata``) for plain imports."""

    try:
        src_path = Path(__file__).resolve().parent / "src"
    except Exception:  # pragma: no cover - defensive guard for unusual import setups
        return

    if not src_path.is_dir():
        return

    src_str = os.fspath(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


try:  # pragma: no cover - executed on interpreter startup
    _ensure_newspaper_cache_directory()
    _ensure_src_on_sys_path()
except Exception:
    # sitecustomize must never block interpreter startup; swallow unexpected errors.
    pass
