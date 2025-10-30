"""Migration executor using event-based tracking and GraphWalker.

This is the new executor that replaces the old 3-table system with a single
event-based system stored in system tables via SystemTableStorage.
"""

from typing import TYPE_CHECKING

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
        self,
        migration: "Migration",
        store: "MetadataStore",
        project: str,
        *,
        dry_run: bool = False,
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
            project: Project name for event tracking
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
            return self._execute_diff_migration(
                migration, store, project, dry_run=dry_run
            )
        elif isinstance(migration, FullGraphMigration):
            return self._execute_full_graph_migration(
                migration, store, project, dry_run=dry_run
            )
        else:
            # CustomMigration - call its execute method directly
            return migration.execute(store, project, dry_run=dry_run)

    def _execute_diff_migration(
        self,
        migration: "DiffMigration",
        store: "MetadataStore",
        project: str,
        dry_run: bool,
    ) -> "MigrationResult":
        """Execute DiffMigration using GraphWalker.

        Args:
            migration: DiffMigration to execute
            store: Metadata store
            project: Project name for event tracking
            dry_run: If True, only validate

        Returns:
            MigrationResult
        """
        # Delegate to the migration's execute method which handles all the logic
        return migration.execute(store, project, dry_run=dry_run)

    def _execute_full_graph_migration(
        self,
        migration: "FullGraphMigration",
        store: "MetadataStore",
        project: str,
        dry_run: bool,
    ) -> "MigrationResult":
        """Execute FullGraphMigration.

        Args:
            migration: FullGraphMigration to execute
            store: Metadata store
            project: Project name for event tracking
            dry_run: If True, only validate

        Returns:
            MigrationResult
        """
        # FullGraphMigration has custom execute logic in the subclass
        # Base implementation is a no-op
        return migration.execute(store, project, dry_run=dry_run)
