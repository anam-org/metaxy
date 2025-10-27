"""System table storage layer for metadata store.

Provides type-safe access to migration system tables using struct-based storage.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

import narwhals as nw
import polars as pl

from metaxy.metadata_store._protocols import MetadataStoreProtocol
from metaxy.models.types import FeatureKey

# System namespace
SYSTEM_NAMESPACE = "metaxy-system"

# System table keys
FEATURE_VERSIONS_KEY = FeatureKey([SYSTEM_NAMESPACE, "feature_versions"])
MIGRATION_EVENTS_KEY = FeatureKey([SYSTEM_NAMESPACE, "migration_events"])
# Note: No migrations table - definitions live in YAML files, only events are stored

# Context variable for suppressing feature_version warning in migrations
_suppress_feature_version_warning: ContextVar[bool] = ContextVar(
    "_suppress_feature_version_warning", default=False
)


@contextmanager
def allow_feature_version_override() -> Iterator[None]:
    """
    Context manager to suppress warnings when writing metadata with pre-existing feature_version.

    This should only be used in migration code where writing historical feature versions
    is intentional and necessary.

    Example:
        >>> with allow_feature_version_override():
        ...     # DataFrame already has feature_version column from migration
        ...     store.write_metadata(MyFeature, df_with_feature_version)
    """
    token = _suppress_feature_version_warning.set(True)
    try:
        yield
    finally:
        _suppress_feature_version_warning.reset(token)


# Common Polars schemas for system tables
# TODO: Migrate to use METAXY_*_COL constants instead of plain names
FEATURE_VERSIONS_SCHEMA = {
    "feature_key": pl.String,
    "feature_version": pl.String,  # TODO: Use METAXY_FEATURE_VERSION_COL
    "recorded_at": pl.Datetime("us"),
    "feature_spec": pl.String,
    "feature_class_path": pl.String,
    "snapshot_version": pl.String,  # TODO: Use METAXY_SNAPSHOT_ID_COL
}

MIGRATION_EVENTS_SCHEMA = {
    "migration_id": pl.String,
    "event_type": pl.String,  # "started", "feature_started", "feature_completed", "completed", "failed"
    "timestamp": pl.Datetime("us"),
    "feature_key": pl.String,  # Empty for migration-level events
    "rows_affected": pl.Int64,
    "error_message": pl.String,  # Empty if no error
}


class SystemTableStorage:
    """Storage layer for migration system tables.

    Provides type-safe access to migration snapshots, migrations, and events.
    Uses struct-based storage (not JSON/bytes) for efficient queries.

    Status is computed at query-time from events (append-only).
    """

    def __init__(self, store: MetadataStoreProtocol):
        """Initialize storage layer.

        Args:
            store: Metadata store to use for system tables
        """
        self.store = store

    # ========== Migrations ==========
    # Note: Migration definitions are stored in YAML files (git), not in the database.
    # Only execution events are stored in DB for tracking progress and state.

    def list_executed_migrations(self) -> list[str]:
        """List all migration IDs that have execution events.

        Returns:
            List of migration IDs that have been started/executed
        """
        lazy = self.store._read_metadata_native(MIGRATION_EVENTS_KEY)

        if lazy is None:
            return []

        df = lazy.select("migration_id").unique().collect().to_polars()
        return df["migration_id"].to_list()

    # ========== Events ==========

    def write_event(
        self,
        migration_id: str,
        event_type: str,
        feature_key: str = "",
        rows_affected: int = 0,
        error_message: str = "",
    ) -> None:
        """Write migration event to system table (append-only).

        Args:
            migration_id: Migration this event belongs to
            event_type: Event type ("started", "feature_started", "feature_completed", "completed", "failed")
            feature_key: Feature key (empty for migration-level events)
            rows_affected: Number of rows affected (for feature events)
            error_message: Error message (empty if no error)
        """
        record = pl.DataFrame(
            {
                "migration_id": [migration_id],
                "event_type": [event_type],
                "timestamp": [datetime.now(timezone.utc)],
                "feature_key": [feature_key],
                "rows_affected": [rows_affected],
                "error_message": [error_message],
            },
            schema=MIGRATION_EVENTS_SCHEMA,
        )
        self.store._write_metadata_impl(MIGRATION_EVENTS_KEY, record)

    def get_migration_events(self, migration_id: str) -> nw.LazyFrame[Any]:
        """Get all events for a migration.

        Args:
            migration_id: Migration ID

        Returns:
            Lazy frame with events sorted by timestamp
        """
        lazy = self.store._read_metadata_native(
            MIGRATION_EVENTS_KEY,
            filters=[nw.col("migration_id") == migration_id],
        )

        if lazy is None:
            # No events yet
            return nw.from_native(pl.DataFrame(schema=MIGRATION_EVENTS_SCHEMA).lazy())

        return lazy.sort("timestamp", descending=False)

    def get_migration_status(self, migration_id: str) -> str:
        """Compute migration status from events at query-time.

        Args:
            migration_id: Migration ID

        Returns:
            Status: "not_started", "in_progress", "completed", "failed"
        """
        events_lazy = self.get_migration_events(migration_id)
        events_df = events_lazy.collect().to_polars()

        if events_df.height == 0:
            return "not_started"

        # Get latest event
        latest_event = events_df.sort("timestamp", descending=True).head(1)
        latest_event_type = latest_event["event_type"][0]

        if latest_event_type == "completed":
            return "completed"
        elif latest_event_type == "failed":
            return "failed"
        elif latest_event_type in ("started", "feature_started", "feature_completed"):
            return "in_progress"

        return "not_started"

    def is_feature_completed(self, migration_id: str, feature_key: str) -> bool:
        """Check if a specific feature completed successfully in a migration.

        Args:
            migration_id: Migration ID
            feature_key: Feature key to check

        Returns:
            True if feature completed without errors
        """
        events_lazy = self.get_migration_events(migration_id)
        events_df = (
            events_lazy.filter(
                (nw.col("feature_key") == feature_key)
                & (nw.col("event_type") == "feature_completed")
                & (nw.col("error_message") == "")
            )
            .collect()
            .to_polars()
        )

        return events_df.height > 0

    def get_completed_features(self, migration_id: str) -> list[str]:
        """Get list of features that completed successfully in a migration.

        Args:
            migration_id: Migration ID

        Returns:
            List of feature keys
        """
        events_lazy = self.get_migration_events(migration_id)
        events_df = (
            events_lazy.filter(
                (nw.col("event_type") == "feature_completed")
                & (nw.col("error_message") == "")
            )
            .collect()
            .to_polars()
        )

        return events_df["feature_key"].unique().to_list()

    def get_failed_features(self, migration_id: str) -> dict[str, str]:
        """Get features that failed in a migration with error messages.

        Args:
            migration_id: Migration ID

        Returns:
            Dict mapping feature key to error message
        """
        events_lazy = self.get_migration_events(migration_id)
        events_df = (
            events_lazy.filter(
                (nw.col("event_type") == "feature_completed")
                & (nw.col("error_message") != "")
            )
            .collect()
            .to_polars()
        )

        result = {}
        for row in events_df.iter_rows(named=True):
            result[row["feature_key"]] = row["error_message"]

        return result
