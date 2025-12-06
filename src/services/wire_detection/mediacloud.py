"""MediaCloud-based wire detection helpers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional
from urllib.parse import urlparse

from mediacloud.api import SearchApi
from mediacloud.error import APIResponseError, MCException

DEFAULT_RATE_PER_MINUTE = 2.0
LOG = logging.getLogger(__name__)


@dataclass
class MediaCloudArticle:
    """Minimal article metadata required for MediaCloud lookups."""

    article_id: str
    source: str
    url: str
    title: str
    extracted_at: Optional[datetime]

    @property
    def host(self) -> str:
        return normalize_host(self.url)


@dataclass
class DetectionResult:
    """Structured result returned by ``MediaCloudDetector``."""

    article: MediaCloudArticle
    query: str
    story_count: int
    matched_story_count: int
    matched_hosts: list[str]
    matched_story_ids: list[str]
    status: str
    queried_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def has_matches(self) -> bool:
        return self.matched_story_count > 0

    def to_metadata(self) -> dict:
        """Return JSON-serialisable metadata for storage in the DB."""

        return {
            "article_id": self.article.article_id,
            "source": self.article.source,
            "query": self.query,
            "story_count": self.story_count,
            "matched_story_count": self.matched_story_count,
            "matched_hosts": list(self.matched_hosts),
            "matched_story_ids": list(self.matched_story_ids),
            "status": self.status,
            "queried_at": self.queried_at.isoformat(),
        }

    def to_wire_payload(self) -> dict:
        """Build a wire attribution payload compatible with ``articles.wire``."""

        payload = {
            "provider": "mediacloud",
            "detection_method": "headline_duplicate",
            "matched_hosts": list(self.matched_hosts),
            "matched_story_ids": list(self.matched_story_ids),
            "matched_story_count": self.matched_story_count,
            "story_count": self.story_count,
            "queried_at": self.queried_at.isoformat(),
            "query": self.query,
        }
        return payload


class RateLimiter:
    """Lightweight token bucket enforcing a minimum call interval."""

    def __init__(self, calls_per_minute: float) -> None:
        if calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be positive")
        self._min_interval = 60.0 / calls_per_minute
        self._last_call: Optional[float] = None

    def wait(self) -> None:
        if self._last_call is None:
            return
        elapsed = time.monotonic() - self._last_call
        sleep_for = self._min_interval - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    def record(self) -> None:
        self._last_call = time.monotonic()


def normalize_host(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
    except ValueError:
        netloc = ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def parse_iso8601(value: str) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_query(title: str) -> str:
    escaped = title.replace('"', '\\"')
    return f'"{escaped}"'


def summarize_matches(
    base: MediaCloudArticle,
    stories: Iterable[dict],
) -> tuple[int, list[str], list[str]]:
    matched_hosts: list[str] = []
    matched_story_ids: list[str] = []
    base_host = base.host
    for story in stories:
        story_url = story.get("url") or ""
        story_host = normalize_host(story_url)
        if not story_host or story_host == base_host:
            continue
        if story_host not in matched_hosts:
            matched_hosts.append(story_host)
        matched_story_ids.append(str(story.get("id")))
    return len(matched_story_ids), matched_hosts, matched_story_ids


class MediaCloudDetector:
    """Perform MediaCloud headline matching with rate limiting."""

    def __init__(
        self,
        search_api: SearchApi,
        *,
        rate_limiter: RateLimiter | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.search_api = search_api
        self.rate_limiter = rate_limiter or RateLimiter(DEFAULT_RATE_PER_MINUTE)
        self.log = logger or LOG

    @classmethod
    def from_token(
        cls,
        token: str,
        *,
        rate_per_minute: float = DEFAULT_RATE_PER_MINUTE,
        logger: logging.Logger | None = None,
    ) -> MediaCloudDetector:
        if not token:
            raise ValueError("MediaCloud API token is required")
        try:
            search_api = SearchApi(token)
        except MCException as exc:  # pragma: no cover - constructor rarely fails
            raise RuntimeError(
                f"Failed to initialise MediaCloud client: {exc}"
            ) from exc
        return cls(
            search_api=search_api,
            rate_limiter=RateLimiter(rate_per_minute),
            logger=logger,
        )

    def detect(self, article: MediaCloudArticle) -> DetectionResult:
        query = build_query(article.title)
        self.log.debug(
            "Submitting MediaCloud query '%s' for article %s", query, article.article_id
        )

        self.rate_limiter.wait()
        stories: list[dict]
        try:
            stories = self._story_list(article, query)
            status = "ok"
        except APIResponseError as exc:
            self.log.warning(
                "MediaCloud API error for article %s: %s", article.article_id, exc
            )
            status = f"api_error:{exc.status_code}"
            stories = []
        except Exception as exc:  # noqa: BLE001
            self.log.exception(
                "Unexpected MediaCloud failure for article %s", article.article_id
            )
            status = f"error:{exc.__class__.__name__}"
            stories = []
        finally:
            self.rate_limiter.record()

        story_count = len(stories)
        matched_count, matched_hosts, matched_story_ids = summarize_matches(
            article, stories
        )
        self.log.debug(
            "MediaCloud returned %d stories, %d matched hosts",
            story_count,
            len(matched_hosts),
        )

        return DetectionResult(
            article=article,
            query=query,
            story_count=story_count,
            matched_story_count=matched_count,
            matched_hosts=matched_hosts,
            matched_story_ids=matched_story_ids,
            status=status,
        )

    def _story_list(
        self,
        article: MediaCloudArticle,
        query: str,
        *,
        page_size: int = 100,
    ) -> list[dict]:
        extracted_at = article.extracted_at
        if extracted_at is None:
            start_date = end_date = datetime.utcnow().date()
        else:
            extracted_at = extracted_at.astimezone(timezone.utc)
            start_date = (extracted_at - timedelta(days=1)).date()
            end_date = (extracted_at + timedelta(days=1)).date()

        stories, _ = self.search_api.story_list(
            query=query,
            start_date=start_date,
            end_date=end_date,
            page_size=page_size,
        )
        return stories


def ensure_timezone(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
