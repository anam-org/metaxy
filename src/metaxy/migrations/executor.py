"""Migration executor using event-based tracking and GraphWalker.

This is the new executor that replaces the old 3-table system with a single
event-based system stored in system tables via SystemTableStorage.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from metaxy.migrations.ops import DataVersionReconciliation

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.metadata_store.system_tables import SystemTableStorage
    from metaxy.migrations.models import (
        DiffMigration,
        FullGraphMigration,
        Migration,
        MigrationResult,
    )


class MigrationExecutor:
    """Executes migrations with event-based progress tracking.

    Uses GraphWalker for topological traversal and SystemTableStorage for
    event logging. Supports resumability after failures.
    """

    def __init__(self, storage: "SystemTableStorage"):
        """Initialize executor.

        Args:
            storage: System table storage for event logging
        """
        self.storage = storage

    def execute(
        self, migration: "Migration", store: "MetadataStore", *, dry_run: bool = False
    ) -> "MigrationResult":
        """Execute migration with event logging and resumability.

        Process:
        1. Log migration_started event
        2. Get features to process from migration
        3. Use GraphWalker to get topological order
        4. For each feature:
           - Check if already completed (resume support)
           - Log feature_started
           - Execute migration logic
           - Log feature_completed/failed
        5. Log migration_completed/failed

        Args:
            migration: Migration to execute
            store: Metadata store to operate on
            dry_run: If True, only validate without executing

        Returns:
            MigrationResult with execution details

        Raises:
            Exception: If migration fails and cannot continue
        """
        # Import here to avoid circular dependency
        from metaxy.migrations.models import DiffMigration, FullGraphMigration

        # Delegate to migration's execute method (which uses this executor internally)
        if isinstance(migration, DiffMigration):
            return self._execute_diff_migration(migration, store, dry_run=dry_run)
        elif isinstance(migration, FullGraphMigration):
            return self._execute_full_graph_migration(migration, store, dry_run=dry_run)
        else:
            # CustomMigration - call its execute method directly
            return migration.execute(store, dry_run=dry_run)

    def _execute_diff_migration(
        self, migration: "DiffMigration", store: "MetadataStore", dry_run: bool
    ) -> "MigrationResult":
        """Execute DiffMigration using GraphWalker.

        Args:
            migration: DiffMigration to execute
            store: Metadata store
            dry_run: If True, only validate

        Returns:
            MigrationResult
        """
        from metaxy.migrations.models import MigrationResult

        start_time = datetime.now(timezone.utc)

        # Note: GraphDiff is not needed for execution
        # It can be computed on-demand via migration.compute_graph_diff(store) if needed

        # Write migration_started event
        if not dry_run:
            self.storage.write_event(migration.migration_id, "started")

        affected_features = []
        errors = {}
        rows_affected_total = 0

        # Get affected features (computed on-demand for DiffMigration)
        affected_features_to_process = migration.get_affected_features(store)

        # Execute for each affected feature in topological order
        for feature_key_str in affected_features_to_process:
            # Check if already completed (resume support)
            if not dry_run and self.storage.is_feature_completed(
                migration.migration_id, feature_key_str
            ):
                affected_features.append(feature_key_str)
                continue

            # Log feature_started
            if not dry_run:
                self.storage.write_event(
                    migration.migration_id,
                    "feature_started",
                    feature_key=feature_key_str,
                )

            try:
                # Execute data version reconciliation for this feature
                op = DataVersionReconciliation()

                rows_affected = op.execute_for_feature(
                    store,
                    feature_key_str,
                    from_snapshot_version=migration.from_snapshot_version,
                    to_snapshot_version=migration.to_snapshot_version,
                    dry_run=dry_run,
                )

                # Log feature_completed
                if not dry_run:
                    self.storage.write_event(
                        migration.migration_id,
                        "feature_completed",
                        feature_key=feature_key_str,
                        rows_affected=rows_affected,
                    )

                affected_features.append(feature_key_str)
                rows_affected_total += rows_affected

            except Exception as e:
                error_msg = str(e)
                errors[feature_key_str] = error_msg

                # Log feature_failed
                if not dry_run:
                    self.storage.write_event(
                        migration.migration_id,
                        "feature_completed",
                        feature_key=feature_key_str,
                        error_message=error_msg,
                    )

                continue

        # Determine status
        if dry_run:
            status = "skipped"
        elif len(errors) == 0:
            status = "completed"
            if not dry_run:
                self.storage.write_event(migration.migration_id, "completed")
        else:
            status = "failed"
            if not dry_run:
                self.storage.write_event(migration.migration_id, "failed")

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        return MigrationResult(
            migration_id=migration.migration_id,
            status=status,
            features_completed=len(affected_features),
            features_failed=len(errors),
            affected_features=affected_features,
            errors=errors,
            rows_affected=rows_affected_total,
            duration_seconds=duration,
            timestamp=start_time,
        )

    def _execute_full_graph_migration(
        self,
        migration: "FullGraphMigration",
        store: "MetadataStore",
        dry_run: bool,
    ) -> "MigrationResult":
        """Execute FullGraphMigration.

        Args:
            migration: FullGraphMigration to execute
            store: Metadata store
            dry_run: If True, only validate

        Returns:
            MigrationResult
        """
        # FullGraphMigration has custom execute logic in the subclass
        # Base implementation is a no-op
        return migration.execute(store, dry_run=dry_run)
