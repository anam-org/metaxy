"""System table storage layer for metadata store.

Provides type-safe access to migration system tables using struct-based storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import narwhals as nw
import polars as pl

from metaxy.metadata_store.exceptions import SystemDataNotFoundError
from metaxy.metadata_store.system import (
    FEATURE_VERSIONS_KEY,
)
from metaxy.metadata_store.system.events import (
    COL_EVENT_TYPE,
    COL_EXECUTION_ID,
    COL_FEATURE_KEY,
    COL_PAYLOAD,
    COL_PROJECT,
    COL_TIMESTAMP,
    Event,
    EventType,
    MigrationStatus,
)
from metaxy.metadata_store.system.keys import EVENTS_KEY
from metaxy.metadata_store.system.models import POLARS_SCHEMAS, FeatureVersionsModel
from metaxy.models.constants import (
    METAXY_FULL_DEFINITION_VERSION,
    METAXY_SNAPSHOT_VERSION,
)
from metaxy.models.feature import FeatureGraph
from metaxy.models.types import FeatureKey, SnapshotPushResult

if TYPE_CHECKING:
    from metaxy.metadata_store import MetadataStore


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

    def __init__(self, store: MetadataStore):
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

        Note:
            The store must already be open when calling this method.
        """
        events = self._read_system_metadata(EVENTS_KEY)

        # Apply project filter only if specified
        if project is not None:
            events = events.filter(nw.col(COL_PROJECT) == project)

        return (
            events.select(COL_EXECUTION_ID)
            .unique()
            .collect()
            .to_polars()[COL_EXECUTION_ID]
            .to_list()
        )

    def write_event(self, event: Event) -> None:
        """Write migration event to system table using typed event models.

        This is the preferred way to write events with full type safety.

        Args:
            event: A typed migration event created via Event classmethods

        Note:
            The store must already be open when calling this method.

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
        self.store.write_metadata(EVENTS_KEY, record)

    def get_migration_events(
        self, migration_id: str, project: str | None = None
    ) -> pl.DataFrame:
        """Get all events for a migration.

        Args:
            migration_id: Migration ID
            project: Optional project name to filter by. If None, returns events for all projects.

        Returns:
            Polars DataFrame with events sorted by timestamp

        Note:
            The store must already be open when calling this method.
        """
        # Read the table first without project filter
        events = self._read_system_metadata(EVENTS_KEY)
        if events is None:
            # Table doesn't exist yet, return empty DataFrame with correct schema
            return pl.DataFrame(schema=POLARS_SCHEMAS[EVENTS_KEY])

        return (
            events.filter(
                nw.col(COL_EXECUTION_ID) == migration_id,
                nw.col(COL_PROJECT) == project,
            )
            .sort(COL_TIMESTAMP, descending=False)
            .collect()
            .to_polars()
        )

    def get_migration_status(
        self,
        migration_id: str,
        project: str | None = None,
        expected_features: list[str] | None = None,
    ) -> MigrationStatus:
        """Compute migration status from events at query-time.

        Args:
            migration_id: Migration ID
            project: Optional project name to filter by. If None, returns status across all projects.
            expected_features: Optional list of feature keys that should be completed.
                If provided, will check that ALL expected features have completed successfully
                before returning COMPLETED status, even if migration_completed event exists.
                This allows detecting when a migration YAML has been modified after completion.

        Returns:
            MigrationStatus enum value
        """

        events_df = self.get_migration_events(migration_id, project=project)

        if events_df.height == 0:
            return MigrationStatus.NOT_STARTED

        # Get latest event
        latest_event = events_df.sort(COL_TIMESTAMP, descending=True).head(1)
        latest_event_type = latest_event[COL_EVENT_TYPE][0]

        # If expected_features is provided, verify ALL features have completed
        # This ensures we detect when operations are added to an already-completed migration
        if expected_features is not None and len(expected_features) > 0:
            completed_features = set(self.get_completed_features(migration_id, project))
            expected_features_set = set(expected_features)

            # Check if all expected features have been completed
            all_features_completed = expected_features_set.issubset(completed_features)

            if not all_features_completed:
                # Some features are missing - migration is not complete
                if latest_event_type in (
                    EventType.MIGRATION_STARTED.value,
                    EventType.FEATURE_MIGRATION_STARTED.value,
                    EventType.FEATURE_MIGRATION_COMPLETED.value,
                    EventType.FEATURE_MIGRATION_FAILED.value,
                ):
                    return MigrationStatus.IN_PROGRESS
                elif latest_event_type == EventType.MIGRATION_FAILED.value:
                    return MigrationStatus.FAILED
                else:
                    # Migration was marked complete but features are missing (YAML was modified)
                    return MigrationStatus.IN_PROGRESS
            # If all features completed, continue with normal status logic below

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
        self,
        migration_id: str,
        project: str | None = None,
        expected_features: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get a comprehensive summary of migration execution status.

        This is a convenience method that returns all migration information
        in a single call, avoiding multiple queries.

        Args:
            migration_id: Migration ID
            project: Optional project name to filter by. If None, returns summary across all projects.
            expected_features: Optional list of feature keys that should be completed.
                If provided, will be used to determine if migration is truly complete.

        Returns:
            Dict containing:
            - status: MigrationStatus enum value
            - completed_features: List of completed feature keys
            - failed_features: Dict mapping failed feature keys to error messages
            - total_features_processed: Count of completed + failed features
        """
        status = self.get_migration_status(migration_id, project, expected_features)
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
        lazy = self._read_system_metadata(EVENTS_KEY)

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
        Note:
            The store must already be open when calling this method.
        """
        lazy = self._read_system_metadata(EVENTS_KEY)
        if lazy is None:
            # Table doesn't exist yet, return empty list
            return []

        # Filter to only completed migrations using narwhals
        completed_events = lazy.filter(nw.col(COL_EVENT_TYPE) == "completed")

        if project is not None:
            completed_events = completed_events.filter(nw.col(COL_PROJECT) == project)

        # Convert to polars LazyFrame and collect
        completed_df = completed_events.to_native().collect()

        if completed_df.height == 0:
            return []

        # Get all events for all migrations at once
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

    def push_graph_snapshot(
        self, tags: dict[str, Any] | None = None
    ) -> SnapshotPushResult:
        """Record all features in graph with a graph snapshot version.

        This should be called during CD (Continuous Deployment) to record what
        feature versions are being deployed. Typically invoked via `metaxy graph push`.

        Records all features in the graph with the same snapshot_version, representing
        a consistent state of the entire feature graph based on code definitions.

        The snapshot_version is a deterministic hash of all feature_version hashes
        in the graph, making it idempotent - calling multiple times with the
        same feature definitions produces the same snapshot_version.

        This method detects three scenarios:
        1. New snapshot (computational changes): No existing rows with this snapshot_version
        2. Metadata-only changes: Snapshot exists but some features have different feature_spec_version
        3. No changes: Snapshot exists with identical feature_spec_versions for all features

        Args:
            tags: Optional dictionary of custom tags to attach to the snapshot
                     (e.g., git commit SHA).

        Note:
            The store must already be open when calling this method.

        Returns: SnapshotPushResult
        """
        tags = tags or {}

        graph = FeatureGraph.get_active()

        # Check if this exact snapshot already exists for this project
        latest_pushed_snapshot = self._read_latest_snapshot_data(graph.snapshot_version)
        current_snapshot_dict = graph.to_snapshot()

        # Convert to DataFrame - need to serialize feature_spec dict to JSON string
        # and add metaxy_snapshot_version and recorded_at columns
        import json
        from datetime import datetime, timezone

        current_snapshot = pl.concat(
            [
                FeatureVersionsModel.model_validate(
                    {
                        "feature_key": k,
                        **{
                            field: (
                                json.dumps(val)
                                if field in ("feature_spec", "feature_schema")
                                else val
                            )
                            for field, val in v.items()
                        },
                        METAXY_SNAPSHOT_VERSION: graph.snapshot_version,
                        "recorded_at": datetime.now(timezone.utc),
                        "tags": json.dumps(tags),
                    }
                ).to_polars()
                for k, v in current_snapshot_dict.items()
            ]
        )

        # Initialize to_push and already_pushed
        to_push = current_snapshot  # Will be updated if snapshot already exists
        already_pushed: bool

        if len(latest_pushed_snapshot) != 0:
            # this snapshot_version HAS been previously pushed
            # let's check for any differences

            already_pushed = True
        else:
            # this snapshot_version has not been previously pushed at all
            # we are safe to push all the features in our graph!
            to_push = current_snapshot
            already_pushed = False

        if len(latest_pushed_snapshot) != 0:
            # let's identify features that have updated definitions since the last push
            # Join full current snapshot with latest pushed (keeping all columns)
            pushed_with_current = current_snapshot.join(
                latest_pushed_snapshot.select(
                    "feature_key",
                    pl.col(METAXY_FULL_DEFINITION_VERSION).alias(
                        f"{METAXY_FULL_DEFINITION_VERSION}_pushed"
                    ),
                ),
                on=["feature_key"],
                how="left",
            )

            to_push = pl.concat(
                [
                    # these are records that for some reason have not been pushed previously
                    pushed_with_current.filter(
                        pl.col(f"{METAXY_FULL_DEFINITION_VERSION}_pushed").is_null()
                    ),
                    # these are the records with actual changes
                    pushed_with_current.filter(
                        pl.col(f"{METAXY_FULL_DEFINITION_VERSION}_pushed").is_not_null()
                    ).filter(
                        pl.col(METAXY_FULL_DEFINITION_VERSION)
                        != pl.col(f"{METAXY_FULL_DEFINITION_VERSION}_pushed")
                    ),
                ]
            ).drop(f"{METAXY_FULL_DEFINITION_VERSION}_pushed")

        if len(to_push) > 0:
            self.store.write_metadata(FEATURE_VERSIONS_KEY, to_push)

        # updated_features only populated when updating existing features
        updated_features = (
            to_push["feature_key"].to_list()
            if len(latest_pushed_snapshot) != 0 and len(to_push) > 0
            else []
        )

        return SnapshotPushResult(
            snapshot_version=graph.snapshot_version,
            already_pushed=already_pushed,
            updated_features=updated_features,
        )

    def _read_system_metadata(self, key: FeatureKey) -> nw.LazyFrame[Any]:
        """Read system metadata.

        System tables are handled specially by MetadataStore.read_metadata - they don't
        require feature plan resolution when current_only=False.

        Note:
            The store must already be open when calling this method.

        Returns:
            LazyFrame if table exists, empty LazyFrame with correct schema if it doesn't
        """
        try:
            # read_metadata handles system tables specially (no feature plan needed)
            return self.store.read_metadata(key, current_only=False)
        except SystemDataNotFoundError:
            return nw.from_native(pl.DataFrame(schema=POLARS_SCHEMAS[key])).lazy()

    def _read_latest_snapshot_data(
        self,
        snapshot_version: str,
    ) -> pl.DataFrame:
        """Read the latest snapshot data for a given snapshot version.

        The same snapshot version may include multiple features as their no-topological metadata such as Pydantic fields or spec.metadata/tags change.
        This method retrieves the latest feature data for each feature pushed to the metadata store.

        Returns:
            Polars DataFrame (materialized) with the latest data. Empty if table doesn't exist or snapshot not found.
        """
        graph = FeatureGraph.get_active()

        # Read system metadata
        sys_meta = self._read_system_metadata(FEATURE_VERSIONS_KEY)

        # Filter the data
        lazy = sys_meta.filter(
            nw.col(METAXY_SNAPSHOT_VERSION) == snapshot_version,
            nw.col("project") == next(iter(graph.features_by_key.values())).project
            if len(graph.features_by_key) > 0
            else "_empty_graph_",
        )

        # Deduplicate using Polars (collect and use native operations)
        return (
            lazy.collect()
            .to_polars()
            .sort("recorded_at", descending=True)
            .unique(subset=["feature_key"], keep="first")
        )

    def read_graph_snapshots(self, project: str | None = None) -> pl.DataFrame:
        """Read recorded graph snapshots from the feature_versions system table.

        Args:
            project: Project name to filter by. If None, returns snapshots from all projects.

        Returns a DataFrame with columns:
        - snapshot_version: Unique identifier for each graph snapshot
        - recorded_at: Timestamp when the snapshot was recorded
        - feature_count: Number of features in this snapshot

        Returns:
            Polars DataFrame with snapshot information, sorted by recorded_at descending

        Raises:
            StoreNotOpenError: If store is not open

        Example:
            ```py
            with store:
                storage = SystemTableStorage(store)
                # Get snapshots for a specific project
                snapshots = storage.read_graph_snapshots(project="my_project")
                latest_snapshot = snapshots[METAXY_SNAPSHOT_VERSION][0]
                print(f"Latest snapshot: {latest_snapshot}")

                # Get snapshots across all projects
                all_snapshots = storage.read_graph_snapshots()
            ```
        """
        # Read system metadata
        versions_lazy = self._read_system_metadata(FEATURE_VERSIONS_KEY)
        if versions_lazy is None:
            # No snapshots recorded yet
            return pl.DataFrame(
                schema={
                    METAXY_SNAPSHOT_VERSION: pl.String,
                    "recorded_at": pl.Datetime("us"),
                    "feature_count": pl.UInt32,
                }
            )

        # Build filters based on project parameter
        if project is not None:
            versions_lazy = versions_lazy.filter(nw.col("project") == project)

        # Materialize
        versions_df = versions_lazy.collect().to_polars()

        if versions_df.height == 0:
            # No snapshots recorded yet
            return pl.DataFrame(
                schema={
                    METAXY_SNAPSHOT_VERSION: pl.String,
                    "recorded_at": pl.Datetime("us"),
                    "feature_count": pl.UInt32,
                }
            )

        # Group by snapshot_version and get earliest recorded_at and count
        snapshots = (
            versions_df.group_by(METAXY_SNAPSHOT_VERSION)
            .agg(
                [
                    pl.col("recorded_at").min().alias("recorded_at"),
                    pl.col("feature_key").count().alias("feature_count"),
                ]
            )
            .sort("recorded_at", descending=True)
        )

        return snapshots

    def read_features(
        self,
        *,
        current: bool = True,
        snapshot_version: str | None = None,
        project: str | None = None,
    ) -> pl.DataFrame:
        """Read feature version information from the feature_versions system table.

        Args:
            current: If True, only return features from the current code snapshot.
                     If False, must provide snapshot_version.
            snapshot_version: Specific snapshot version to filter by. Required if current=False.
            project: Project name to filter by. Defaults to None.

        Returns:
            Polars DataFrame with columns from FEATURE_VERSIONS_SCHEMA:
            - feature_key: Feature identifier
            - feature_version: Version hash of the feature
            - recorded_at: When this version was recorded
            - feature_spec: JSON serialized feature specification
            - feature_class_path: Python import path to the feature class
            - snapshot_version: Graph snapshot this feature belongs to

        Raises:
            StoreNotOpenError: If store is not open
            ValueError: If current=False but no snapshot_version provided

        Examples:
            ```py
            # Get features from current code
            with store:
                storage = SystemTableStorage(store)
                features = storage.read_features(current=True)
                print(f"Current graph has {len(features)} features")
            ```

            ```py
            # Get features from a specific snapshot
            with store:
                storage = SystemTableStorage(store)
                features = storage.read_features(current=False, snapshot_version="abc123")
                for row in features.iter_rows(named=True):
                    print(f"{row['feature_key']}: {row['metaxy_feature_version']}")
            ```
        """
        if not current and snapshot_version is None:
            raise ValueError("Must provide snapshot_version when current=False")

        if current:
            # Get current snapshot from active graph
            graph = FeatureGraph.get_active()
            snapshot_version = graph.snapshot_version

        # Read system metadata
        versions_lazy = self._read_system_metadata(FEATURE_VERSIONS_KEY)
        if versions_lazy is None:
            # No features recorded yet
            return pl.DataFrame(schema=POLARS_SCHEMAS[FEATURE_VERSIONS_KEY])

        # Build filters
        filters = [nw.col(METAXY_SNAPSHOT_VERSION) == snapshot_version]
        if project is not None:
            filters.append(nw.col("project") == project)

        for f in filters:
            versions_lazy = versions_lazy.filter(f)

        # Materialize
        versions_df = versions_lazy.collect().to_polars()

        return versions_df

    def load_graph_from_snapshot(
        self,
        snapshot_version: str,
        project: str | None = None,
        *,
        class_path_overrides: dict[str, str] | None = None,
        force_reload: bool = False,
    ) -> FeatureGraph:
        """Load and reconstruct a FeatureGraph from a stored snapshot.

        This is a convenience method that encapsulates the pattern of:

        1. Reading feature metadata for a snapshot

        2. Building the snapshot data dictionary

        3. Reconstructing the FeatureGraph from snapshot data

        Args:
            snapshot_version: The snapshot version to load
            project: Optional project name to filter by
            class_path_overrides: Optional dict mapping feature_key to new class path
                                  for features that have been moved/renamed
            force_reload: If True, force reimport of feature classes even if cached

        Returns:
            Reconstructed FeatureGraph

        Raises:
            ValueError: If no features found for the snapshot version
            ImportError: If feature classes cannot be imported at their recorded paths

        Note:
            The store must already be open when calling this method.

        Example:
            ```python
            with store:
                storage = SystemTableStorage(store)
                graph = storage.load_graph_from_snapshot(
                    snapshot_version="abc123",
                    project="my_project"
                )
                print(f"Loaded {len(graph.features_by_key)} features")
            ```
        """
        import json

        # Read features for this snapshot
        features_df = self.read_features(
            current=False,
            snapshot_version=snapshot_version,
            project=project,
        )

        if features_df.height == 0:
            raise ValueError(
                f"No features recorded for snapshot {snapshot_version}"
                + (f" in project {project}" if project else "")
            )

        # Build snapshot data dict for FeatureGraph.from_snapshot()
        snapshot_data = {
            row["feature_key"]: {
                "feature_spec": json.loads(row["feature_spec"])
                if isinstance(row["feature_spec"], str)
                else row["feature_spec"],
                "feature_class_path": row["feature_class_path"],
                "metaxy_feature_version": row["metaxy_feature_version"],
            }
            for row in features_df.iter_rows(named=True)
        }

        # Reconstruct graph from snapshot
        return FeatureGraph.from_snapshot(
            snapshot_data,
            class_path_overrides=class_path_overrides,
            force_reload=force_reload,
        )
