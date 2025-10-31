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
    "project": pl.String,
    "feature_key": pl.String,
    "feature_version": pl.String,  # TODO: Use METAXY_FEATURE_VERSION_COL
    "feature_spec_version": pl.String,  # Hash of complete BaseFeatureSpec (all properties)
    "feature_tracking_version": pl.String,  # Hash of feature_spec_version + project (for migration detection)
    "recorded_at": pl.Datetime("us"),
    "feature_spec": pl.String,  # Full serialized BaseFeatureSpec
    "feature_class_path": pl.String,
    "snapshot_version": pl.String,  # TODO: Use METAXY_SNAPSHOT_ID_COL
}

MIGRATION_EVENTS_SCHEMA = {
    "project": pl.String,
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

    def list_executed_migrations(self, project: str | None = None) -> list[str]:
        """List all migration IDs that have execution events.

        Args:
            project: Optional project name to filter by. If None, returns migrations for all projects.

        Returns:
            List of migration IDs that have been started/executed
        """
        # First try to read without filter to check if table exists
        lazy = self.store._read_metadata_native(MIGRATION_EVENTS_KEY)

        if lazy is None:
            return []

        # Handle backward compatibility and project filtering
        if project is not None:
            try:
                # Try to filter by project (new schema)
                lazy = lazy.filter(nw.col("project") == project)
            except Exception:
                # If project column doesn't exist (old schema), return all migrations
                # This maintains backward compatibility with existing tables
                pass
        # If project is None, we don't filter - return all migrations across all projects

        df = lazy.select("migration_id").unique().collect().to_polars()
        return df["migration_id"].to_list()

    # ========== Events ==========

    def write_event(
        self,
        migration_id: str,
        event_type: str,
        project: str,
        feature_key: str = "",
        rows_affected: int = 0,
        error_message: str = "",
    ) -> None:
        """Write migration event to system table (append-only).

        Args:
            migration_id: Migration this event belongs to
            event_type: Event type ("started", "feature_started", "feature_completed", "completed", "failed")
            project: Project name for isolation
            feature_key: Feature key (empty for migration-level events)
            rows_affected: Number of rows affected (for feature events)
            error_message: Error message (empty if no error)
        """
        record = pl.DataFrame(
            {
                "project": [project],
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

    def get_migration_events(
        self, migration_id: str, project: str | None = None
    ) -> nw.LazyFrame[Any]:
        """Get all events for a migration.

        Args:
            migration_id: Migration ID
            project: Optional project name to filter by. If None, returns events for all projects.

        Returns:
            Lazy frame with events sorted by timestamp
        """
        # Read the table first without project filter
        lazy = self.store._read_metadata_native(
            MIGRATION_EVENTS_KEY,
            filters=[nw.col("migration_id") == migration_id],
        )

        if lazy is None:
            # No events yet
            return nw.from_native(pl.DataFrame(schema=MIGRATION_EVENTS_SCHEMA).lazy())

        lazy = lazy.filter(nw.col("project") == project)
        return lazy.sort("timestamp", descending=False)

    def get_migration_status(
        self, migration_id: str, project: str | None = None
    ) -> str:
        """Compute migration status from events at query-time.

        Args:
            migration_id: Migration ID
            project: Optional project name to filter by. If None, returns status across all projects.

        Returns:
            Status: "not_started", "in_progress", "completed", "failed"
        """

        events_lazy = self.get_migration_events(migration_id, project=project)
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

    def is_feature_completed(
        self, migration_id: str, feature_key: str, project: str | None = None
    ) -> bool:
        """Check if a specific feature completed successfully in a migration.

        Args:
            migration_id: Migration ID
            feature_key: Feature key to check
            project: Optional project name to filter by. If None, checks across all projects.

        Returns:
            True if feature completed without errors
        """
        events_lazy = self.get_migration_events(migration_id, project)
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

    def get_completed_features(
        self, migration_id: str, project: str | None = None
    ) -> list[str]:
        """Get list of features that completed successfully in a migration.

        Args:
            migration_id: Migration ID
            project: Optional project name to filter by. If None, returns features for all projects.

        Returns:
            List of feature keys
        """
        events_lazy = self.get_migration_events(migration_id, project=project)
        events_df = (
            events_lazy.filter(
                (nw.col("event_type") == "feature_completed")
                & (nw.col("error_message") == "")
            )
            .collect()
            .to_polars()
        )

        return events_df["feature_key"].unique().to_list()

    def get_failed_features(
        self, migration_id: str, project: str | None = None
    ) -> dict[str, str]:
        """Get features that failed in a migration with error messages.

        Args:
            migration_id: Migration ID
            project: Optional project name to filter by. If None, returns features for all projects.

        Returns:
            Dict mapping feature key to error message
        """
        events_lazy = self.get_migration_events(migration_id, project=project)
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

    # ========== Convenience Methods for Reading Migration Data ==========

    def read_migration_events(
        self, project: str | None = None, migration_id: str | None = None
    ) -> nw.LazyFrame[Any]:
        """Read all migration events, optionally filtered by project and/or migration ID.

        Args:
            project: Optional project name to filter by. If None, returns events for all projects.
            migration_id: Optional migration ID to filter by. If None, returns events for all migrations.

        Returns:
            Lazy frame with migration events
        """
        lazy = self.store._read_metadata_native(MIGRATION_EVENTS_KEY)

        if lazy is None:
            # No events yet
            return nw.from_native(pl.DataFrame(schema=MIGRATION_EVENTS_SCHEMA).lazy())

        # Apply filters if specified
        if migration_id is not None:
            lazy = lazy.filter(nw.col("migration_id") == migration_id)

        if project is not None:
            try:
                # Try to filter by project (new schema)
                lazy = lazy.filter(nw.col("project") == project)
            except Exception:
                # If project column doesn't exist (old schema), we can't filter by project
                pass

        return lazy.sort("timestamp", descending=False)

    def read_migration_progress(
        self, project: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """Read migration progress across all migrations.

        Args:
            project: Optional project name to filter by. If None, returns progress for all projects.

        Returns:
            Dict mapping migration_id to progress information including:
            - status: "not_started", "in_progress", "completed", "failed"
            - completed_features: List of completed feature keys
            - failed_features: Dict of failed feature keys to error messages
            - total_rows_affected: Total rows affected across all features
        """
        # Get all migration IDs
        migration_ids = self.list_executed_migrations(project)

        progress = {}
        for mid in migration_ids:
            events_lazy = self.read_migration_events(project=project, migration_id=mid)
            events_df = events_lazy.collect().to_polars()

            if events_df.height == 0:
                continue

            # Get latest event for status
            latest_event = events_df.sort("timestamp", descending=True).head(1)
            latest_event_type = latest_event["event_type"][0]

            if latest_event_type == "completed":
                status = "completed"
            elif latest_event_type == "failed":
                status = "failed"
            elif latest_event_type in (
                "started",
                "feature_started",
                "feature_completed",
            ):
                status = "in_progress"
            else:
                status = "not_started"

            # Get completed features
            completed_df = events_df.filter(
                (events_df["event_type"] == "feature_completed")
                & (events_df["error_message"] == "")
            )
            completed_features = completed_df["feature_key"].unique().to_list()

            # Get failed features
            failed_df = events_df.filter(
                (events_df["event_type"] == "feature_completed")
                & (events_df["error_message"] != "")
            )
            failed_features = {}
            for row in failed_df.iter_rows(named=True):
                failed_features[row["feature_key"]] = row["error_message"]

            # Calculate total rows affected
            total_rows = events_df.filter(
                events_df["event_type"] == "feature_completed"
            )["rows_affected"].sum()

            progress[mid] = {
                "status": status,
                "completed_features": completed_features,
                "failed_features": failed_features,
                "total_rows_affected": total_rows or 0,
            }

        return progress

    def read_applied_migrations(
        self, project: str | None = None
    ) -> list[dict[str, Any]]:
        """Read all applied (completed) migrations with their details.

        Args:
            project: Optional project name to filter by. If None, returns migrations for all projects.

        Returns:
            List of dicts containing migration details for completed migrations:
            - migration_id: Migration ID
            - project: Project name (if available)
            - completed_at: Timestamp when migration completed
            - features_count: Number of features affected
            - rows_affected: Total rows affected
        """
        lazy = self.store._read_metadata_native(MIGRATION_EVENTS_KEY)

        if lazy is None:
            return []

        # Filter to only completed migrations
        completed_events = lazy.filter(nw.col("event_type") == "completed")

        if project is not None:
            try:
                completed_events = completed_events.filter(nw.col("project") == project)
            except Exception:
                # If project column doesn't exist, we can't filter
                pass

        completed_df = completed_events.collect().to_polars()

        if completed_df.height == 0:
            return []

        applied = []
        for row in completed_df.iter_rows(named=True):
            mid = row["migration_id"]

            # Get all events for this migration
            all_events_lazy = self.read_migration_events(
                project=row.get("project"), migration_id=mid
            )
            all_events = all_events_lazy.collect().to_polars()

            # Count features and rows
            feature_events = all_events.filter(
                all_events["event_type"] == "feature_completed"
            )
            features_count = feature_events["feature_key"].n_unique()
            rows_affected = feature_events["rows_affected"].sum() or 0

            applied.append(
                {
                    "migration_id": mid,
                    "project": row.get("project", "unknown"),
                    "completed_at": row["timestamp"],
                    "features_count": features_count,
                    "rows_affected": rows_affected,
                }
            )

        return applied
