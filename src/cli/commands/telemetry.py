"""Telemetry command module for querying extraction performance data."""

import logging

from src.utils.comprehensive_telemetry import ComprehensiveExtractionTelemetry

logger = logging.getLogger(__name__)


def add_telemetry_parser(subparsers):
    """Add telemetry command parser to CLI."""
    telemetry_parser = subparsers.add_parser(
        "telemetry",
        help="Query extraction performance telemetry",
    )

    # Add subcommands
    telemetry_subparsers = telemetry_parser.add_subparsers(
        dest="telemetry_command",
        help="Telemetry analysis commands",
    )

    # HTTP errors summary
    errors_parser = telemetry_subparsers.add_parser(
        "errors",
        help="Show HTTP error summary",
    )
    errors_parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to look back (default: 7)",
    )

    # Method effectiveness
    methods_parser = telemetry_subparsers.add_parser(
        "methods",
        help="Show extraction method effectiveness",
    )
    methods_parser.add_argument(
        "--publisher",
        type=str,
        help="Filter by specific publisher",
    )

    # Publisher stats
    telemetry_subparsers.add_parser(
        "publishers",
        help="Show per-publisher performance statistics",
    )

    # Bot protection performance
    protection_parser = telemetry_subparsers.add_parser(
        "protection",
        help="Show success rates by bot protection type",
    )
    protection_parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Lookback window in hours (default: 24)",
    )

    # Unblock proxy outcomes
    unblock_parser = telemetry_subparsers.add_parser(
        "unblock",
        help="Show unblock proxy outcomes by host",
    )
    unblock_parser.add_argument(
        "--hours",
        type=int,
        default=12,
        help="Lookback window in hours (default: 12)",
    )

    # Field extraction analysis
    fields_parser = telemetry_subparsers.add_parser(
        "fields",
        help="Show field-level extraction success by method",
    )
    fields_parser.add_argument(
        "--publisher",
        type=str,
        help="Filter by specific publisher",
    )
    fields_parser.add_argument(
        "--method",
        type=str,
        choices=["newspaper4k", "beautifulsoup", "selenium"],
        help="Filter by specific extraction method",
    )

    # Set the handler function
    telemetry_parser.set_defaults(func=handle_telemetry_command)


def handle_telemetry_command(args) -> int:
    """Handle telemetry command with subcommands."""
    try:
        telemetry = ComprehensiveExtractionTelemetry()

        if args.telemetry_command == "errors":
            return _show_http_errors(telemetry, args.days)
        elif args.telemetry_command == "methods":
            return _show_method_effectiveness(telemetry, args.publisher)
        elif args.telemetry_command == "publishers":
            return _show_publisher_stats(telemetry)
        elif args.telemetry_command == "fields":
            return _show_field_extraction(
                telemetry,
                args.publisher,
                getattr(args, "method", None),
            )
        elif args.telemetry_command == "protection":
            return _show_protection_stats(telemetry, args.hours)
        elif args.telemetry_command == "unblock":
            return _show_unblock_summary(telemetry, args.hours)
        else:
            print(
                "Please specify a telemetry subcommand: errors, methods, "
                "publishers, fields, protection, or unblock",
            )
            return 1

    except Exception as e:
        logger.error(f"Telemetry command failed: {e}")
        return 1


def _show_http_errors(telemetry, days: int) -> int:
    """Show HTTP error summary."""
    print(f"\n🚨 HTTP Error Summary (Last {days} days)")
    print("=" * 50)

    errors = telemetry.get_error_summary(days)

    if not errors:
        print("No HTTP errors recorded in the specified time period.")
        return 0

    # Group by error type
    error_types = {}
    for error in errors:
        error_type = error["error_type"]
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(error)

    for error_type, type_errors in error_types.items():
        print(f"\\n📊 {error_type.upper()} ERRORS:")
        print("-" * 30)

        # Sort by count descending
        type_errors.sort(key=lambda x: x["count"], reverse=True)

        for error in type_errors[:10]:  # Show top 10
            print(
                f"  {error['host'][:40]:40} | "
                f"Status: {error['status_code']} | "
                f"Count: {error['count']:3d} | "
                f"Last: {error['last_seen'][:19]}"
            )

    print(f"\n📈 Total error records: {len(errors)}")
    return 0


# End of telemetry CLI helpers


def _show_method_effectiveness(
    telemetry,
    publisher: str | None = None,
) -> int:
    """Show extraction method effectiveness."""
    filter_text = f" (Publisher: {publisher})" if publisher else ""
    print(f"\n⚡ Extraction Method Effectiveness{filter_text}")
    print("=" * 60)

    methods = telemetry.get_method_effectiveness(publisher)

    if not methods:
        print("No method effectiveness data available.")
        return 0

    print(f"{'Method':<15} {'Count':<8} {'Success Rate':<12} {'Avg Duration':<12}")
    print("-" * 60)

    for method in methods:
        method_name = method["successful_method"] or "Unknown"
        count = method["count"]
        success_rate = method["success_rate"] * 100 if method["success_rate"] else 0
        avg_duration = method["avg_duration"] / 1000 if method["avg_duration"] else 0

        print(
            f"{method_name:<15} {count:<8} "
            f"{success_rate:<11.1f}% {avg_duration:<11.1f}s"
        )

    print(f"\\n📊 Total method records: {len(methods)}")
    return 0


def _show_publisher_stats(telemetry) -> int:
    """Show per-publisher performance statistics."""
    print("\n📰 Publisher Performance Statistics")
    print("=" * 80)

    publishers = telemetry.get_publisher_stats()

    if not publishers:
        print("No publisher statistics available.")
        return 0

    # Group by publisher
    pub_stats = {}
    for stat in publishers:
        pub = stat["publisher"] or "Unknown"
        if pub not in pub_stats:
            pub_stats[pub] = []
        pub_stats[pub].append(stat)

    for publisher, stats in pub_stats.items():
        print(f"\\n🏢 {publisher}")
        print("-" * 60)

        total_attempts = sum(s["total_attempts"] for s in stats)
        total_successful = sum(s["successful"] for s in stats)
        success_rate = (
            (total_successful / total_attempts * 100) if total_attempts > 0 else 0
        )

        print(
            "Overall: "
            f"{total_attempts} attempts, {total_successful} successful "
            f"({success_rate:.1f}%)"
        )

        print(
            f"\n{'Host':<30} {'Attempts':<10} {'Success':<10} "
            f"{'Success%':<10} {'Avg Time':<10}"
        )
        print("." * 80)

        # Sort by attempts descending
        stats.sort(key=lambda x: x["total_attempts"], reverse=True)

        for stat in stats[:10]:  # Show top 10 hosts
            host = stat["host"][:29] if stat["host"] else "Unknown"
            attempts = stat["total_attempts"]
            successful = stat["successful"]
            host_success_rate = successful / attempts * 100 if attempts > 0 else 0
            avg_duration = (
                stat["avg_duration_ms"] / 1000 if stat["avg_duration_ms"] else 0
            )

            print(
                f"{host:<30} {attempts:<10} "
                f"{successful:<10} {host_success_rate:<9.1f}% "
                f"{avg_duration:<9.1f}s"
            )

    print(f"\\n📊 Total publisher/host combinations: {len(publishers)}")
    return 0


def _show_field_extraction(
    telemetry,
    publisher: str | None = None,
    method: str | None = None,
) -> int:
    """Show field-level extraction success by method."""
    filter_text = ""
    if publisher:
        filter_text += f" (Publisher: {publisher})"
    if method:
        filter_text += f" (Method: {method})"

    print(f"\\n🎯 Field Extraction Analysis{filter_text}")
    print("=" * 80)

    # Get field extraction data from the database
    field_data = telemetry.get_field_extraction_stats(publisher, method)

    if not field_data:
        print("No field extraction data available.")
        return 0

    # Display field success rates by method
    print(
        f"\n{'Method':<15} {'Title':<8} {'Author':<8} "
        f"{'Content':<8} {'Date':<8} {'Metadata':<10} {'Count':<8}"
    )
    print("-" * 80)

    for data in field_data:
        method_name = data["method"] or "Unknown"
        title_rate = data["title_success_rate"] * 100
        author_rate = data["author_success_rate"] * 100
        content_rate = data["content_success_rate"] * 100
        date_rate = data["date_success_rate"] * 100
        metadata_rate = data.get("metadata_success_rate", 0.0) * 100
        count = data["count"]

        print(
            f"{method_name:<15} {title_rate:<7.1f}% "
            f"{author_rate:<7.1f}% {content_rate:<7.1f}% "
            f"{date_rate:<7.1f}% {metadata_rate:<9.1f}% {count:<8}"
        )

    # Show overall field success across all methods
    print("\\n📊 Field Extraction Summary:")
    total_extractions = sum(data["count"] for data in field_data)
    if total_extractions > 0:
        overall_title = (
            sum(data["title_success_rate"] * data["count"] for data in field_data)
            / total_extractions
            * 100
        )
        overall_author = (
            sum(data["author_success_rate"] * data["count"] for data in field_data)
            / total_extractions
            * 100
        )
        overall_content = (
            sum(data["content_success_rate"] * data["count"] for data in field_data)
            / total_extractions
            * 100
        )
        overall_date = (
            sum(data["date_success_rate"] * data["count"] for data in field_data)
            / total_extractions
            * 100
        )
        overall_metadata = (
            sum(
                data.get("metadata_success_rate", 0) * data["count"]
                for data in field_data
            )
            / total_extractions
            * 100
        )

        print(f"Title Success:   {overall_title:.1f}%")
        print(f"Author Success:  {overall_author:.1f}%")
        print(f"Content Success: {overall_content:.1f}%")
        print(f"Date Success:    {overall_date:.1f}%")
        print(f"Metadata Success: {overall_metadata:6.1f}%")
        print(f"Total Records:   {total_extractions}")

    return 0


def _show_protection_stats(telemetry, hours: int) -> int:
    """Display success metrics grouped by bot protection type."""

    stats = telemetry.get_protection_success_stats(hours)

    print(f"\n🛡️  Bot Protection Performance (last {hours}h)")
    print("=" * 90)

    if not stats:
        print("No extraction attempts recorded in the selected window.")
        return 0

    header = (
        f"{'Protection':<20} {'Attempts':<9} {'Success%':<9} "
        f"{'Unblock (succ/att)':<20} {'Selenium (succ/att)':<23}"
    )
    print(header)
    print("-" * len(header))

    for stat in stats:
        protection = stat["protection_type"] or "unknown"
        attempts = stat["attempts"]
        success_pct = stat["success_rate"] * 100
        unblock_attempts = stat["unblock_attempts"]
        unblock_successes = stat["unblock_successes"]
        unblock_rate = stat["unblock_success_rate"] * 100
        selenium_attempts = stat["selenium_attempts"]
        selenium_successes = stat["selenium_successes"]
        selenium_rate = stat["selenium_success_rate"] * 100

        unblock_text = (
            f"{unblock_successes}/{unblock_attempts} ({unblock_rate:>5.1f}%)"
            if unblock_attempts
            else "0/0 ( 0.0%)"
        )
        selenium_text = (
            f"{selenium_successes}/{selenium_attempts} ({selenium_rate:>5.1f}%)"
            if selenium_attempts
            else "0/0 ( 0.0%)"
        )

        print(
            f"{protection:<20} {attempts:<9d} {success_pct:>7.1f}% "
            f"{unblock_text:<20} {selenium_text:<23}"
        )

    return 0


def _show_unblock_summary(telemetry, hours: int) -> int:
    """Display unblock proxy outcomes by host and protection type."""

    summary = telemetry.get_unblock_proxy_outcomes(hours)
    host_stats = summary.get("host_stats", [])
    protection_stats = summary.get("protection_stats", [])
    window_start = summary.get("window_start")

    print(f"\n🕵️  Unblock Proxy Outcomes (last {hours}h)")
    print("=" * 90)

    if window_start:
        print(f"Window start: {window_start.isoformat()}\n")

    if not host_stats:
        print("No unblock proxy attempts recorded in the selected window.")
        return 0

    print("Protection families:")
    for stat in protection_stats:
        attempts = stat["attempts"]
        success_pct = stat["success_rate"] * 100
        print(
            f"  - {stat['protection_type']}: {attempts} attempts, "
            f"{stat['successes']} successes, {stat['challenges']} challenges, "
            f"{stat['failures']} failures (success {success_pct:.1f}%)"
        )

    print("\nTop hosts:")
    print(
        f"{'Host':<35} {'Protection':<15} {'Attempts':<9} "
        f"{'Success':<8} {'Challenge':<10} {'Failure':<8} {'Last Seen':<20}"
    )
    print("-" * 110)

    for stat in host_stats[:20]:
        last_seen = stat["last_seen"]
        last_seen_str = last_seen.strftime("%Y-%m-%d %H:%M") if last_seen else "n/a"
        print(
            f"{stat['host']:<35} {stat['protection_type']:<15} "
            f"{stat['attempts']:<9d} {stat['successes']:<8d} "
            f"{stat['challenges']:<10d} {stat['failures']:<8d} {last_seen_str:<20}"
        )

    print(
        "\nChallenge counts correspond to proxy blocks where the article "
        "was left in 'article' status for retry."
    )
    return 0


# Telemetry CLI helpers end here
