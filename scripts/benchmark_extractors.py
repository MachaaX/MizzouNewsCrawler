#!/usr/bin/env python3
"""Benchmark existing extractor pipeline against ``mcmetadata`` on a URL sample.

This utility fetches each URL once, then runs both the current
``ContentExtractor`` implementation and the ``mcmetadata`` extractor against
the same HTML.  It records timing, success rates, content lengths, and basic
metadata availability so we can compare performance and recall before deciding
on deeper integration.

Usage examples
--------------

Compare a handful of URLs listed in ``sample_urls.txt``::

	poetry run python scripts/benchmark_extractors.py sample_urls.txt \
		--limit 50 --output results/benchmark.csv

Run against a CSV exported from our database::

	poetry run python scripts/benchmark_extractors.py data/articles.csv \
		--csv --column url --sleep 0.25

The script reports aggregate metrics to STDOUT and optionally writes detailed
per-URL rows to CSV for further analysis.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional

import requests

# Ensure repository root is on the import path when running via ``python``
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from src.crawler import ContentExtractor  # noqa: E402  (heavy import, defered)

logger = logging.getLogger(__name__)


try:  # pragma: no cover - optional dependency we want to check at runtime
	import mcmetadata
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
	raise SystemExit(
		"mcmetadata is required for this benchmark. Run from the repository root "
		"(vendored package) or install `mediacloud-metadata` manually."
	) from exc


@dataclass
class FetchResult:
	"""Container for fetch responses."""

	html: Optional[str]
	status_code: Optional[int]
	response_bytes: Optional[int]
	error: Optional[str]
	duration_sec: float


@dataclass
class ExtractionResult:
	"""Normalized extraction outcome."""

	success: bool
	title: Optional[str]
	title_chars: int
	content_chars: int
	has_author: bool
	authors: Optional[list[str]]
	publish_date: Optional[str]
	text_extraction_method: Optional[str]
	duration_sec: float
	error: Optional[str]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"input",
		type=Path,
		help="Path to a newline-delimited text file or CSV containing URLs.",
	)
	parser.add_argument(
		"--csv",
		action="store_true",
		help="Treat the input file as CSV (default: newline separated URLs).",
	)
	parser.add_argument(
		"--column",
		default="url",
		help="Column name to read URLs from when --csv is supplied (default: url).",
	)
	parser.add_argument(
		"--limit",
		type=int,
		help="Maximum number of URLs to process (default: all).",
	)
	parser.add_argument(
		"--offset",
		type=int,
		default=0,
		help="Number of URLs to skip before processing (default: 0).",
	)
	parser.add_argument(
		"--min-chars",
		type=int,
		default=200,
		help="Minimum character count to consider text extraction a success (default: 200).",
	)
	parser.add_argument(
		"--timeout",
		type=float,
		default=20.0,
		help="HTTP timeout in seconds for fetching HTML (default: 20).",
	)
	parser.add_argument(
		"--sleep",
		type=float,
		default=0.0,
		help="Optional delay between URL fetches to avoid rate-limits (seconds).",
	)
	parser.add_argument(
		"--output",
		type=Path,
		help="Optional path to write per-URL benchmark results (.csv).",
	)
	parser.add_argument(
		"--include-other-metadata",
		action="store_true",
		help="Request mcmetadata to include extended metadata (authors, etc.).",
	)
	parser.add_argument(
		"--user-agent",
		default=(
			"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
			"AppleWebKit/537.36 (KHTML, like Gecko) "
			"Chrome/121.0.0.0 Safari/537.36"
		),
		help="Custom User-Agent string for HTTP fetches.",
	)
	parser.add_argument(
		"--log-level",
		default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR"],
		help="Logging verbosity (default: INFO).",
	)
	return parser.parse_args()


def load_urls(
	path: Path,
	*,
	use_csv: bool,
	column: str,
	limit: Optional[int],
	offset: int,
) -> list[str]:
	"""Read URLs from text or CSV file, applying optional limit."""

	urls: list[str] = []

	if not path.exists():
		raise FileNotFoundError(f"Input file not found: {path}")

	if use_csv:
		with path.open("r", encoding="utf-8", newline="") as handle:
			reader = csv.DictReader(handle)
			if column not in reader.fieldnames:
				raise ValueError(
					f"Column '{column}' not found in CSV header {reader.fieldnames}"
				)
			for index, row in enumerate(reader):
				if index < offset:
					continue
				url = (row.get(column) or "").strip()
				if not url:
					continue
				urls.append(url)
				if limit and len(urls) >= limit:
					break
	else:
		with path.open("r", encoding="utf-8") as handle:
			for index, line in enumerate(handle):
				if index < offset:
					continue
				url = line.strip()
				if not url or url.startswith("#"):
					continue
				urls.append(url)
				if limit and len(urls) >= limit:
					break

	return urls


def fetch_html(
	session: requests.Session, url: str, timeout: float
) -> FetchResult:
	"""Fetch URL content capturing timing and errors."""

	start = time.perf_counter()
	try:
		response = session.get(url, timeout=timeout, allow_redirects=True)
		duration = time.perf_counter() - start
		status_code = response.status_code
		content = response.text
		size = len(response.content)
		if response.status_code >= 400:
			error = f"HTTP {response.status_code}"
			logger.debug("Fetch failed %s: %s", url, error)
			return FetchResult(None, status_code, size, error, duration)
		if not content.strip():
			return FetchResult(None, status_code, size, "Empty response", duration)
		return FetchResult(content, status_code, size, None, duration)
	except Exception as exc:  # pragma: no cover - network failures
		duration = time.perf_counter() - start
		logger.debug("Fetch error %s: %s", url, exc)
		return FetchResult(None, None, None, str(exc), duration)


def run_crawler_extractor(
	extractor: ContentExtractor,
	url: str,
	html: str,
	*,
	min_chars: int,
) -> ExtractionResult:
	"""Execute current pipeline extractor with shared HTML."""

	t0 = time.perf_counter()
	try:
		payload = extractor.extract_content(url, html)
		duration = time.perf_counter() - t0
		if not payload:
			return ExtractionResult(False, None, 0, 0, False, None, None, None, duration, "Empty payload")

		title = payload.get("title")
		content = payload.get("content")
		author = payload.get("author")
		publish_date = payload.get("publish_date")
		authors_list: Optional[list[str]] = None
		if isinstance(author, str) and author.strip():
			authors_list = [author.strip()]
		elif isinstance(author, (list, tuple)):
			authors_list = [str(a).strip() for a in author if str(a).strip()]

		content_len = len(content.strip()) if isinstance(content, str) else 0
		title_len = len(title.strip()) if isinstance(title, str) else 0

		success = content_len >= min_chars
		return ExtractionResult(
			success=success,
			title=title,
			title_chars=title_len,
			content_chars=content_len,
			has_author=bool(author),
			authors=authors_list,
			publish_date=publish_date,
			text_extraction_method=(
				payload.get("extraction_methods", {}).get("content")
				or payload.get("primary_method")
			),
			duration_sec=duration,
			error=None,
		)
	except Exception as exc:  # pragma: no cover - runtime errors we want to record
		duration = time.perf_counter() - t0
		logger.debug("Crawler extractor failed %s: %s", url, exc)
		return ExtractionResult(False, None, 0, 0, False, None, None, None, duration, str(exc))


def run_mcmetadata(
	url: str,
	html: str,
	*,
	min_chars: int,
	include_other_metadata: bool,
) -> ExtractionResult:
	"""Execute mcmetadata extraction using provided HTML."""

	t0 = time.perf_counter()
	stats_accumulator = dict.fromkeys(mcmetadata.STAT_NAMES, 0)
	try:
		result = mcmetadata.extract(
			url=url,
			html_text=html,
			include_other_metadata=include_other_metadata,
			stats_accumulator=stats_accumulator,
		)
		duration = time.perf_counter() - t0
		text = result.get("text_content")
		title = result.get("article_title")
		authors_raw = result.get("other", {}).get("authors") if result.get("other") else None
		if isinstance(authors_raw, str):
			authors = [authors_raw.strip()]
		elif isinstance(authors_raw, (list, tuple)):
			authors = [str(a).strip() for a in authors_raw if str(a).strip()]
		else:
			authors = None
		pub_date = result.get("publication_date")

		content_len = len(text.strip()) if isinstance(text, str) else 0
		title_len = len(title.strip()) if isinstance(title, str) else 0

		success = content_len >= min_chars

		return ExtractionResult(
			success=success,
			title=title,
			title_chars=title_len,
			content_chars=content_len,
			has_author=bool(authors),
			authors=authors,
			publish_date=str(pub_date) if pub_date else None,
			text_extraction_method=result.get("text_extraction_method"),
			duration_sec=duration,
			error=None,
		)
	except Exception as exc:  # pragma: no cover - we want these recorded
		duration = time.perf_counter() - t0
		logger.debug("mcmetadata failed %s: %s", url, exc)
		return ExtractionResult(False, None, 0, 0, False, None, None, None, duration, str(exc))


def write_output(path: Path, rows: Iterable[dict[str, object]]) -> None:
	"""Write detail rows to CSV."""

	fieldnames = [
		"url",
		"status_code",
		"fetch_error",
		"fetch_duration_sec",
		"response_bytes",
		"crawler_success",
		"crawler_duration_sec",
		"crawler_content_chars",
		"crawler_title_chars",
		"crawler_title",
		"crawler_has_author",
		"crawler_authors",
		"crawler_publish_date",
		"crawler_method",
		"crawler_error",
		"mcmetadata_success",
		"mcmetadata_duration_sec",
		"mcmetadata_content_chars",
		"mcmetadata_title_chars",
		"mcmetadata_title",
		"mcmetadata_has_author",
		"mcmetadata_authors",
		"mcmetadata_publish_date",
		"mcmetadata_method",
		"mcmetadata_error",
	]

	path.parent.mkdir(parents=True, exist_ok=True)

	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def summarize(results: list[dict[str, object]]) -> None:
	"""Print a concise summary of collected metrics."""

	total = len(results)
	if total == 0:
		print("No URLs processed.")
		return

	fetch_failures = sum(1 for r in results if r["fetch_error"])

	crawler_successes = [
		r["crawler_success"] for r in results if r["crawler_success"] is not None
	]
	mcmetadata_successes = [
		r["mcmetadata_success"]
		for r in results
		if r["mcmetadata_success"] is not None
	]

	both_success = sum(
		1
		for r in results
		if r["crawler_success"] and r["mcmetadata_success"]
	)

	crawler_durations = [
		r["crawler_duration_sec"]
		for r in results
		if r["crawler_duration_sec"] is not None
	]
	mcmetadata_durations = [
		r["mcmetadata_duration_sec"]
		for r in results
		if r["mcmetadata_duration_sec"] is not None
	]

	def pct(values: Iterable[bool]) -> float:
		values = list(values)
		if not values:
			return 0.0
		return 100.0 * sum(1 for v in values if v) / len(values)

	print("Processed URLs:", total)
	print(f"Fetch failures: {fetch_failures} ({fetch_failures / total:.1%})")
	print(
		f"Crawler success rate: {pct(crawler_successes):.1f}% (n={len(crawler_successes)})"
	)
	print(
		f"mcmetadata success rate: {pct(mcmetadata_successes):.1f}% (n={len(mcmetadata_successes)})"
	)
	print(f"Both succeeded: {both_success} ({both_success / total:.1%})")

	if crawler_durations:
		print(
			f"Crawler avg duration: {mean(crawler_durations):.2f}s "
			f"(median approx not computed)"
		)
	if mcmetadata_durations:
		print(
			f"mcmetadata avg duration: {mean(mcmetadata_durations):.2f}s "
			f"(median approx not computed)"
		)


def main() -> None:
	args = parse_args()
	logging.basicConfig(
		level=getattr(logging, args.log_level),
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)

	urls = load_urls(
		args.input,
		use_csv=args.csv,
		column=args.column,
		limit=args.limit,
		offset=args.offset,
	)
	if not urls:
		raise SystemExit("No URLs to process. Check your input file/filters.")

	session = requests.Session()
	session.headers.update({"User-Agent": args.user_agent})

	extractor = ContentExtractor()

	rows: list[dict[str, object]] = []

	for index, url in enumerate(urls, start=1):
		logger.info("[%d/%d] Fetching %s", index, len(urls), url)
		fetch_result = fetch_html(session, url, args.timeout)

		row: dict[str, object] = {
			"url": url,
			"status_code": fetch_result.status_code,
			"fetch_error": fetch_result.error,
			"fetch_duration_sec": round(fetch_result.duration_sec, 3),
			"response_bytes": fetch_result.response_bytes,
			"crawler_success": None,
			"crawler_duration_sec": None,
			"crawler_content_chars": None,
			"crawler_title_chars": None,
			"crawler_title": None,
			"crawler_has_author": None,
			"crawler_authors": None,
			"crawler_publish_date": None,
			"crawler_method": None,
			"crawler_error": None,
			"mcmetadata_success": None,
			"mcmetadata_duration_sec": None,
			"mcmetadata_content_chars": None,
			"mcmetadata_title_chars": None,
			"mcmetadata_title": None,
			"mcmetadata_has_author": None,
			"mcmetadata_authors": None,
			"mcmetadata_publish_date": None,
			"mcmetadata_method": None,
			"mcmetadata_error": None,
		}

		html = fetch_result.html
		if html:
			crawler_result = run_crawler_extractor(
				extractor,
				url,
				html,
				min_chars=args.min_chars,
			)
			row.update(
				{
					"crawler_success": crawler_result.success,
					"crawler_duration_sec": round(crawler_result.duration_sec, 3),
					"crawler_content_chars": crawler_result.content_chars,
					"crawler_title_chars": crawler_result.title_chars,
					"crawler_title": crawler_result.title,
					"crawler_has_author": crawler_result.has_author,
					"crawler_authors": "; ".join(crawler_result.authors)
					if crawler_result.authors
					else None,
					"crawler_publish_date": crawler_result.publish_date,
					"crawler_method": crawler_result.text_extraction_method,
					"crawler_error": crawler_result.error,
				}
			)

			mcmetadata_result = run_mcmetadata(
				url,
				html,
				min_chars=args.min_chars,
				include_other_metadata=args.include_other_metadata,
			)
			row.update(
				{
					"mcmetadata_success": mcmetadata_result.success,
					"mcmetadata_duration_sec": round(
						mcmetadata_result.duration_sec, 3
					),
					"mcmetadata_content_chars": mcmetadata_result.content_chars,
					"mcmetadata_title_chars": mcmetadata_result.title_chars,
					"mcmetadata_title": mcmetadata_result.title,
					"mcmetadata_has_author": mcmetadata_result.has_author,
					"mcmetadata_authors": "; ".join(mcmetadata_result.authors)
					if mcmetadata_result.authors
					else None,
					"mcmetadata_publish_date": mcmetadata_result.publish_date,
					"mcmetadata_method": mcmetadata_result.text_extraction_method,
					"mcmetadata_error": mcmetadata_result.error,
				}
			)
		else:
			logger.warning("Skipping extraction for %s due to fetch failure", url)

		rows.append(row)

		if args.sleep:
			time.sleep(args.sleep)

	summarize(rows)

	if args.output:
		write_output(args.output, rows)
		logger.info("Wrote detailed results to %s", args.output)


if __name__ == "__main__":
	main()
