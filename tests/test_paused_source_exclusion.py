"""
Test suite for paused source exclusion from discovery.

Verifies that sources with status='paused' are excluded from
get_sources_to_process() and not processed during discovery runs.
"""

import json

import pytest

from src.crawler.discovery import NewsDiscovery
from src.models import create_tables
from src.models.database import DatabaseManager, safe_execute


class TestPausedSourceExclusion:
    """Test that paused sources are excluded from discovery."""

    @pytest.fixture
    def mock_discovery(self, tmp_path):
        """Create a mock NewsDiscovery instance with SQLite for testing."""
        db_path = tmp_path / "test_paused_exclusion.db"
        database_url = f"sqlite:///{db_path}"

        # Create discovery instance
        discovery = NewsDiscovery(database_url=database_url)

        # Create ORM tables
        db_manager = DatabaseManager(database_url)
        create_tables(db_manager.engine)
        yield discovery, db_manager
        db_manager.close()

    def test_paused_sources_excluded_from_discovery(self, mock_discovery):
        """Test that sources with status='paused' are excluded from get_sources_to_process."""
        discovery, db_manager = mock_discovery

        # Insert test sources: one active, one paused, one with NULL status
        with db_manager.engine.begin() as conn:
            safe_execute(
                conn,
                """
                INSERT INTO sources (
                    id, host, host_norm, canonical_name, status,
                    rss_consecutive_failures, rss_transient_failures,
                    no_effective_methods_consecutive
                )
                VALUES
                    (:id1, :host1, :host_norm1, :name1, :status1, 0, '[]', 0),
                    (:id2, :host2, :host_norm2, :name2, :status2, 0, '[]', 0),
                    (:id3, :host3, :host_norm3, :name3, :status3, 0, '[]', 0)
                """,
                {
                    "id1": "active-source-1",
                    "host1": "active-site.com",
                    "host_norm1": "active-site.com",
                    "name1": "Active Site",
                    "status1": "active",
                    "id2": "paused-source-1",
                    "host2": "paused-site.com",
                    "host_norm2": "paused-site.com",
                    "name2": "Paused Site",
                    "status2": "paused",
                    "id3": "null-status-source-1",
                    "host3": "null-status-site.com",
                    "host_norm3": "null-status-site.com",
                    "name3": "Null Status Site",
                    "status3": None,
                },
            )

        # Get sources to process
        sources_df, stats = discovery.get_sources_to_process(
            dataset_label=None, limit=None, due_only=False
        )

        # Verify paused source is excluded
        source_ids = sources_df["id"].tolist()
        assert "active-source-1" in source_ids, "Active source should be included"
        assert (
            "null-status-source-1" in source_ids
        ), "Null status source should be included"
        assert "paused-source-1" not in source_ids, "Paused source should be excluded"

        # Verify total count
        assert len(sources_df) == 2, "Should return 2 sources (active + null status)"

    def test_paused_source_with_reason_excluded(self, mock_discovery):
        """Test that paused sources with a reason are excluded."""
        discovery, db_manager = mock_discovery

        # Insert paused source with reason
        with db_manager.engine.begin() as conn:
            safe_execute(
                conn,
                """
                INSERT INTO sources (
                    id, host, host_norm, canonical_name, status,
                    paused_reason, paused_at,
                    rss_consecutive_failures, rss_transient_failures,
                    no_effective_methods_consecutive
                )
                VALUES (
                    :id, :host, :host_norm, :name, :status,
                    :reason, :paused_at, 0, '[]', 0
                )
                """,
                {
                    "id": "paused-kprs",
                    "host": "www.kprs.com",
                    "host_norm": "www.kprs.com",
                    "name": "Carter Broadcast Group (KPRS/KPRT)",
                    "status": "paused",
                    "reason": "Manually paused - syndicated content",
                    "paused_at": "2025-12-19T00:00:00",
                },
            )

        # Get sources to process
        sources_df, stats = discovery.get_sources_to_process(
            dataset_label=None, limit=None, due_only=False
        )

        # Verify paused source is excluded
        source_ids = sources_df["id"].tolist()
        assert "paused-kprs" not in source_ids, "Paused KPRS source should be excluded"

    def test_multiple_paused_sources_all_excluded(self, mock_discovery):
        """Test that multiple paused sources are all excluded."""
        discovery, db_manager = mock_discovery

        # Insert multiple sources with different statuses
        with db_manager.engine.begin() as conn:
            for i in range(5):
                status = "paused" if i % 2 == 0 else "active"
                safe_execute(
                    conn,
                    """
                    INSERT INTO sources (
                        id, host, host_norm, canonical_name, status,
                        rss_consecutive_failures, rss_transient_failures,
                        no_effective_methods_consecutive
                    )
                    VALUES (:id, :host, :host_norm, :name, :status, 0, '[]', 0)
                    """,
                    {
                        "id": f"source-{i}",
                        "host": f"site-{i}.com",
                        "host_norm": f"site-{i}.com",
                        "name": f"Site {i}",
                        "status": status,
                    },
                )

        # Get sources to process
        sources_df, stats = discovery.get_sources_to_process(
            dataset_label=None, limit=None, due_only=False
        )

        # Verify counts: 5 sources total, 3 paused (0, 2, 4), 2 active (1, 3)
        assert len(sources_df) == 2, "Should return 2 active sources"

        # Verify only active sources are included
        source_ids = sources_df["id"].tolist()
        assert "source-1" in source_ids, "Active source-1 should be included"
        assert "source-3" in source_ids, "Active source-3 should be included"
        assert "source-0" not in source_ids, "Paused source-0 should be excluded"
        assert "source-2" not in source_ids, "Paused source-2 should be excluded"
        assert "source-4" not in source_ids, "Paused source-4 should be excluded"

    def test_host_filter_respects_paused_status(self, mock_discovery):
        """Test that host_filter still respects paused status."""
        discovery, db_manager = mock_discovery

        # Insert active and paused sources with same host pattern
        with db_manager.engine.begin() as conn:
            safe_execute(
                conn,
                """
                INSERT INTO sources (
                    id, host, host_norm, canonical_name, status,
                    rss_consecutive_failures, rss_transient_failures,
                    no_effective_methods_consecutive
                )
                VALUES
                    (:id1, :host1, :host_norm1, :name1, :status1, 0, '[]', 0),
                    (:id2, :host2, :host_norm2, :name2, :status2, 0, '[]', 0)
                """,
                {
                    "id1": "active-test",
                    "host1": "test.com",
                    "host_norm1": "test.com",
                    "name1": "Test Active",
                    "status1": "active",
                    "id2": "paused-test",
                    "host2": "test.org",
                    "host_norm2": "test.org",
                    "name2": "Test Paused",
                    "status2": "paused",
                },
            )

        # Get sources with host_filter (should still exclude paused)
        sources_df, stats = discovery.get_sources_to_process(
            dataset_label=None, limit=None, due_only=False, host_filter="test.com"
        )

        # Should only get the active one with matching host
        assert len(sources_df) == 1, "Should return 1 source"
        assert sources_df.iloc[0]["id"] == "active-test"

        # Try filtering for paused host - should get nothing
        sources_df_paused, stats_paused = discovery.get_sources_to_process(
            dataset_label=None, limit=None, due_only=False, host_filter="test.org"
        )

        assert (
            len(sources_df_paused) == 0
        ), "Paused source should be excluded even with host_filter"

    def test_city_county_filters_respect_paused_status(self, mock_discovery):
        """Test that city/county filters still exclude paused sources."""
        discovery, db_manager = mock_discovery

        # Insert sources with city/county metadata
        with db_manager.engine.begin() as conn:
            safe_execute(
                conn,
                """
                INSERT INTO sources (
                    id, host, host_norm, canonical_name, city, county, status,
                    rss_consecutive_failures, rss_transient_failures,
                    no_effective_methods_consecutive
                )
                VALUES
                    (:id1, :host1, :host_norm1, :name1, :city1, :county1, :status1, 0, '[]', 0),
                    (:id2, :host2, :host_norm2, :name2, :city2, :county2, :status2, 0, '[]', 0)
                """,
                {
                    "id1": "active-boone",
                    "host1": "boone-active.com",
                    "host_norm1": "boone-active.com",
                    "name1": "Boone Active",
                    "city1": "Columbia",
                    "county1": "Boone",
                    "status1": "active",
                    "id2": "paused-boone",
                    "host2": "boone-paused.com",
                    "host_norm2": "boone-paused.com",
                    "name2": "Boone Paused",
                    "city2": "Columbia",
                    "county2": "Boone",
                    "status2": "paused",
                },
            )

        # Get sources by county filter
        sources_df, stats = discovery.get_sources_to_process(
            dataset_label=None, limit=None, due_only=False, county_filter="Boone"
        )

        # Should only get active source
        assert len(sources_df) == 1, "Should return 1 active source in Boone County"
        assert sources_df.iloc[0]["id"] == "active-boone"
        assert (
            "paused-boone" not in sources_df["id"].tolist()
        ), "Paused Boone source should be excluded"
