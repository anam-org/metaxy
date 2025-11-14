"""System table storage layer for metadata store.

Provides type-safe access to migration system tables using struct-based storage.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import narwhals as nw
import polars as pl

from metaxy.metadata_store._protocols import MetadataStoreProtocol
from metaxy.metadata_store.system.events import (
    COL_EVENT_TYPE,
    COL_EXECUTION_ID,
    COL_FEATURE_KEY,
    COL_PAYLOAD,
    COL_PROJECT,
    COL_TIMESTAMP,
    EVENTS_SCHEMA,
    Event,
    EventType,
    MigrationStatus,
)
from metaxy.metadata_store.system.keys import EVENTS_KEY

# Context variable for suppressing feature_version warning in migrations
_suppress_feature_version_warning: ContextVar[bool] = ContextVar(
    "_suppress_feature_version_warning", default=False
)


@contextmanager
def allow_feature_version_override() -> Iterator[None]:
    """
    Context manager to suppress warnings when writing metadata with pre-existing metaxy_feature_version.

    This should only be used in migration code where writing historical feature versions
    is intentional and necessary.

    Example:
        ```py
        with allow_feature_version_override():
            # DataFrame already has metaxy_feature_version column from migration
            store.write_metadata(MyFeature, df_with_feature_version)
        ```
    """
    token = _suppress_feature_version_warning.set(True)
    try:
        yield
    finally:
        _suppress_feature_version_warning.reset(token)


class SystemTableStorage:
    """Storage layer for migration system tables.

    Provides type-safe access to migration snapshots, migrations, and events.
    Uses struct-based storage (not JSON/bytes) for efficient queries.

    Status is computed at query-time from events (append-only).

    Usage:
        ```python
        with SystemTableStorage(store) as storage:
            storage.write_event(Event.migration_started(...))
        ```
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
        with self.store:
            lazy = self.store.read_metadata_in_store(EVENTS_KEY)

            if lazy is None:
                return []

            if project is not None:
                lazy = lazy.filter(nw.col(COL_PROJECT) == project)

            # Select, dedupe, convert to polars, then collect
            df = lazy.select(COL_EXECUTION_ID).unique().to_native().collect()
            return df[COL_EXECUTION_ID].to_list()

    # ========== Events ==========

    def write_event(self, event: Event) -> None:
        """Write migration event to system table using typed event models.

        This is the preferred way to write events with full type safety.

        Args:
            event: A typed migration event created via Event classmethods

        Example:
            ```python
            storage.write_event(
                Event.migration_started(project="my_project", migration_id="m001")
            )

            storage.write_event(
                Event.feature_completed(
                    project="my_project",
                    migration_id="m001",
                    feature_key="feature/a",
                    rows_affected=100,
                )
            )
            ```
        """
        record = event.to_polars()
        with self.store:
            # Write directly to implementation - system tables don't need feature validation
            self.store._write_metadata_impl(EVENTS_KEY, record)

    def get_migration_events(
        self, migration_id: str, project: str | None = None
    ) -> pl.DataFrame:
        """Get all events for a migration.

        Args:
            migration_id: Migration ID
            project: Optional project name to filter by. If None, returns events for all projects.

        Returns:
            Polars DataFrame with events sorted by timestamp
        """
        with self.store:
            # Read the table first without project filter
            lazy = self.store.read_metadata_in_store(
                EVENTS_KEY,
                filters=[nw.col(COL_EXECUTION_ID) == migration_id],
            )

            if lazy is None:
                # No events yet
                return pl.DataFrame(schema=EVENTS_SCHEMA)

            lazy = lazy.filter(nw.col(COL_PROJECT) == project)
            # Convert to Polars DataFrame
            return lazy.sort(COL_TIMESTAMP, descending=False).collect().to_polars()

    def get_migration_status(
        self, migration_id: str, project: str | None = None
    ) -> MigrationStatus:
        """Compute migration status from events at query-time.

        Args:
            migration_id: Migration ID
            project: Optional project name to filter by. If None, returns status across all projects.

        Returns:
            MigrationStatus enum value
        """

        events_df = self.get_migration_events(migration_id, project=project)

        if events_df.height == 0:
            return MigrationStatus.NOT_STARTED

        # Get latest event
        latest_event = events_df.sort(COL_TIMESTAMP, descending=True).head(1)
        latest_event_type = latest_event[COL_EVENT_TYPE][0]

        if latest_event_type == EventType.MIGRATION_COMPLETED.value:
            return MigrationStatus.COMPLETED
        elif latest_event_type == EventType.MIGRATION_FAILED.value:
            return MigrationStatus.FAILED
        elif latest_event_type in (
            EventType.MIGRATION_STARTED.value,
            EventType.FEATURE_MIGRATION_STARTED.value,
            EventType.FEATURE_MIGRATION_COMPLETED.value,
            EventType.FEATURE_MIGRATION_FAILED.value,
        ):
            return MigrationStatus.IN_PROGRESS

        return MigrationStatus.NOT_STARTED

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
        events_df = self.get_migration_events(migration_id, project)

        # Filter and check for completed events without errors
        events_df = (
            events_df.filter(
                (pl.col(COL_FEATURE_KEY) == feature_key)
                & (
                    pl.col(COL_EVENT_TYPE)
                    == EventType.FEATURE_MIGRATION_COMPLETED.value
                )
            )
            .with_columns(
                pl.col(COL_PAYLOAD)
                .str.json_path_match("$.error_message")
                .alias("error_message")
            )
            .filter(pl.col("error_message").is_null() | (pl.col("error_message") == ""))
        )

        # Check if any completed event has no error
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
        events_df = self.get_migration_events(migration_id, project=project)

        # Filter and extract completed features
        events_df = (
            events_df.filter(
                pl.col(COL_EVENT_TYPE) == EventType.FEATURE_MIGRATION_COMPLETED.value
            )
            .with_columns(
                pl.col(COL_PAYLOAD)
                .str.json_path_match("$.error_message")
                .alias("error_message")
            )
            .filter(pl.col("error_message").is_null() | (pl.col("error_message") == ""))
            .select(COL_FEATURE_KEY)
            .unique()
        )

        return events_df[COL_FEATURE_KEY].to_list()

    def get_failed_features(
        self, migration_id: str, project: str | None = None
    ) -> dict[str, str]:
        """Get features that failed in a migration with error messages.

        Only returns features whose LATEST event is a failure. If a feature
        failed and then succeeded on retry, it won't be included here.

        Args:
            migration_id: Migration ID
            project: Optional project name to filter by. If None, returns features for all projects.

        Returns:
            Dict mapping feature key to error message
        """
        events_df = self.get_migration_events(migration_id, project=project)

        if events_df.height == 0:
            return {}

        # Get completed features (these succeeded, even if they failed before)
        completed_features = set(self.get_completed_features(migration_id, project))

        # Filter for failed events, excluding features that later completed
        failed_events = (
            events_df.filter(
                pl.col(COL_EVENT_TYPE) == EventType.FEATURE_MIGRATION_FAILED.value
            )
            .with_columns(
                pl.col(COL_PAYLOAD)
                .str.json_path_match("$.error_message")
                .alias("error_message")
            )
            # Get latest failed event per feature
            .sort(COL_TIMESTAMP, descending=True)
            .group_by(COL_FEATURE_KEY, maintain_order=True)
            .agg([pl.col("error_message").first().alias("error_message")])
            # Exclude features that eventually completed
            .filter(~pl.col(COL_FEATURE_KEY).is_in(list(completed_features)))
            .select([COL_FEATURE_KEY, "error_message"])
        )

        # Convert to dict
        return dict(
            zip(
                failed_events[COL_FEATURE_KEY].to_list(),
                failed_events["error_message"].to_list(),
            )
        )

    def get_migration_summary(
        self, migration_id: str, project: str | None = None
    ) -> dict[str, Any]:
        """Get a comprehensive summary of migration execution status.

        This is a convenience method that returns all migration information
        in a single call, avoiding multiple queries.

        Args:
            migration_id: Migration ID
            project: Optional project name to filter by. If None, returns summary across all projects.

        Returns:
            Dict containing:
            - status: MigrationStatus enum value
            - completed_features: List of completed feature keys
            - failed_features: Dict mapping failed feature keys to error messages
            - total_features_processed: Count of completed + failed features
        """
        status = self.get_migration_status(migration_id, project)
        completed = self.get_completed_features(migration_id, project)
        failed = self.get_failed_features(migration_id, project)

        return {
            "status": status,
            "completed_features": completed,
            "failed_features": failed,
            "total_features_processed": len(completed) + len(failed),
        }

    # ========== Convenience Methods for Reading Migration Data ==========

    def read_migration_events(
        self, project: str | None = None, migration_id: str | None = None
    ) -> pl.DataFrame:
        """Read all migration events, optionally filtered by project and/or migration ID.

        Args:
            project: Optional project name to filter by. If None, returns events for all projects.
            migration_id: Optional migration ID to filter by. If None, returns events for all migrations.

        Returns:
            Polars DataFrame with migration events
        """
        with self.store:
            lazy = self.store.read_metadata_in_store(EVENTS_KEY)

            if lazy is None:
                # No events yet
                return pl.DataFrame(schema=EVENTS_SCHEMA)

            # Apply filters if specified
            if migration_id is not None:
                lazy = lazy.filter(nw.col(COL_EXECUTION_ID) == migration_id)

            if project is not None:
                lazy = lazy.filter(nw.col(COL_PROJECT) == project)

            # Convert to Polars DataFrame
            return lazy.sort(COL_TIMESTAMP, descending=False).collect().to_polars()

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
            events_df = self.read_migration_events(project=project, migration_id=mid)

            if events_df.height == 0:
                continue

            # Get latest event for status
            latest_event = events_df.sort(COL_TIMESTAMP, descending=True).head(1)
            latest_event_type = latest_event[COL_EVENT_TYPE][0]

            if latest_event_type == "completed":
                status = "completed"
            elif latest_event_type == "failed":
                status = "failed"
            elif latest_event_type in (
                "started",
                "feature_started",
                EventType.FEATURE_MIGRATION_COMPLETED.value,
            ):
                status = "in_progress"
            else:
                status = "not_started"

            # Get completed and failed features using JSON path (polars operations on collected data)
            feature_events = events_df.filter(
                events_df[COL_EVENT_TYPE] == EventType.FEATURE_MIGRATION_COMPLETED.value
            ).with_columns(
                [
                    pl.col(COL_PAYLOAD)
                    .str.json_path_match("$.error_message")
                    .alias("error_message"),
                    pl.col(COL_PAYLOAD)
                    .str.json_path_match("$.rows_affected")
                    .cast(pl.Int64)
                    .fill_null(0)
                    .alias("rows_affected"),
                ]
            )

            # Split into completed and failed
            completed_df = feature_events.filter(
                pl.col("error_message").is_null() | (pl.col("error_message") == "")
            )
            failed_df = feature_events.filter(
                pl.col("error_message").is_not_null() & (pl.col("error_message") != "")
            )

            completed_features = completed_df[COL_FEATURE_KEY].unique().to_list()
            failed_features = dict(
                zip(
                    failed_df[COL_FEATURE_KEY].to_list(),
                    failed_df["error_message"].to_list(),
                )
            )
            total_rows = int(feature_events["rows_affected"].sum() or 0)

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
        with self.store:
            lazy = self.store.read_metadata_in_store(EVENTS_KEY)

            if lazy is None:
                return []

            # Filter to only completed migrations using narwhals
            completed_events = lazy.filter(nw.col(COL_EVENT_TYPE) == "completed")

            if project is not None:
                completed_events = completed_events.filter(
                    nw.col(COL_PROJECT) == project
                )

            # Convert to polars LazyFrame and collect
            completed_df = completed_events.to_native().collect()

            if completed_df.height == 0:
                return []

        # Get all events for all migrations at once (this uses with self.store internally)
        all_events = self.read_migration_events(project=project)

        # Extract rows_affected from payload using JSON path (polars operations)
        feature_events = all_events.filter(
            all_events[COL_EVENT_TYPE] == EventType.FEATURE_MIGRATION_COMPLETED.value
        ).with_columns(
            pl.col(COL_PAYLOAD)
            .str.json_path_match("$.rows_affected")
            .cast(pl.Int64)
            .fill_null(0)
            .alias("rows_affected")
        )

        # Group by execution_id to get aggregated stats
        migration_stats = feature_events.group_by(COL_EXECUTION_ID).agg(
            [
                pl.col(COL_FEATURE_KEY).n_unique().alias("features_count"),
                pl.col("rows_affected").sum().alias("rows_affected"),
            ]
        )

        # Join with completed events to get project and timestamp
        result_df = completed_df.join(
            migration_stats, on=COL_EXECUTION_ID, how="left"
        ).select(
            [
                COL_EXECUTION_ID,
                COL_PROJECT,
                pl.col(COL_TIMESTAMP).alias("completed_at"),
                pl.col("features_count").fill_null(0),
                pl.col("rows_affected").fill_null(0).cast(pl.Int64),
            ]
        )

        # Convert to list of dicts
        return result_df.to_dicts()
