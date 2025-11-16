"""Migration executor using event-based tracking.

This is the new executor that replaces the old 3-table system with a single
event-based system stored in system tables via SystemTableStorage.
"""

import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.metadata_store.system import SystemTableStorage
    from metaxy.migrations.models import (
        DiffMigration,
        FullGraphMigration,
        Migration,
        MigrationResult,
    )

from metaxy.migrations.models import OperationConfig


class MigrationExecutor:
    """Executes migrations with event-based progress tracking.

    Uses FeatureGraph.topological_sort_features() for topological traversal
    and SystemTableStorage for event logging. Supports resumability after failures.
    """

    def __init__(self, storage: "SystemTableStorage"):
        """Initialize executor.

        Args:
            storage: System table storage for event logging
        """
        self.storage = storage

    def _find_root_causes(
        self,
        failed_deps: list[str],
        errors: dict[str, str],
    ) -> list[str]:
        """Find the root cause features (features with actual errors, not skipped).

        Args:
            failed_deps: List of direct failed dependencies
            errors: Dict mapping feature keys to error messages

        Returns:
            List of root cause feature keys (features that had actual errors)
        """
        root_causes = []
        for dep in failed_deps:
            if dep not in errors:
                continue
            error_msg = errors[dep]
            # If this dependency was skipped, recursively find its root causes
            if error_msg.startswith("Skipped due to failed dependencies:"):
                # Extract the dependencies from the error message
                deps_part = error_msg.split(":", 1)[1].strip()
                transitive_deps = [d.strip() for d in deps_part.split(",")]
                # Recursively find root causes
                root_causes.extend(self._find_root_causes(transitive_deps, errors))
            else:
                # This is an actual error, not a skip
                root_causes.append(dep)
        return list(
            dict.fromkeys(root_causes)
        )  # Remove duplicates while preserving order

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
        3. Sort features topologically using FeatureGraph.topological_sort_features()
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
            # Custom migration subclass - call its execute method directly
            return migration.execute(store, project, dry_run=dry_run)

    def _execute_diff_migration(
        self,
        migration: "DiffMigration",
        store: "MetadataStore",
        project: str,
        dry_run: bool,
    ) -> "MigrationResult":
        """Execute DiffMigration with topological sorting.

        Args:
            migration: DiffMigration to execute
            store: Metadata store
            project: Project name for event tracking
            dry_run: If True, only validate

        Returns:
            MigrationResult
        """
        import logging
        from datetime import datetime, timezone

        from metaxy.metadata_store.system import Event
        from metaxy.migrations.ops import DataVersionReconciliation
        from metaxy.models.feature import FeatureGraph
        from metaxy.models.types import FeatureKey

        logger = logging.getLogger(__name__)

        start_time = datetime.now(timezone.utc)

        # Log migration started
        if not dry_run:
            self.storage.write_event(
                Event.migration_started(
                    project=project, migration_id=migration.migration_id
                )
            )

        affected_features_list = []
        errors = {}
        skipped = {}  # Track skipped features separately
        rows_affected_total = 0

        # Get affected features (computed on-demand)
        affected_features_to_process = migration.get_affected_features(store, project)

        # Execute operations (currently only DataVersionReconciliation is supported)
        if len(migration.operations) == 1 and isinstance(
            migration.operations[0], DataVersionReconciliation
        ):
            # DataVersionReconciliation applies to all affected features
            op = migration.operations[0]

            for feature_key_str in affected_features_to_process:
                # Check if already completed (resume support)
                if not dry_run and self.storage.is_feature_completed(
                    migration.migration_id, feature_key_str, project
                ):
                    affected_features_list.append(feature_key_str)
                    continue

                # Check if any upstream dependencies failed in this migration run
                feature_key_obj = FeatureKey(feature_key_str.split("/"))
                graph = FeatureGraph.get_active()
                plan = graph.get_feature_plan(feature_key_obj)

                if plan.deps:
                    failed_deps = [
                        dep.key.to_string()
                        for dep in plan.deps
                        if dep.key.to_string() in errors
                        or dep.key.to_string() in skipped
                    ]

                    if failed_deps:
                        # Find root causes (features with actual errors, not just skipped)
                        root_causes = self._find_root_causes(
                            failed_deps, {**errors, **skipped}
                        )
                        error_msg = f"Skipped due to failed dependencies: {', '.join(root_causes)}"
                        skipped[feature_key_str] = error_msg

                        # Log as failed
                        if not dry_run:
                            self.storage.write_event(
                                Event.feature_failed(
                                    project=project,
                                    migration_id=migration.migration_id,
                                    feature_key=feature_key_str,
                                    error_message=error_msg,
                                )
                            )
                        continue

                # Log feature started
                if not dry_run:
                    self.storage.write_event(
                        Event.feature_started(
                            project=project,
                            migration_id=migration.migration_id,
                            feature_key=feature_key_str,
                        )
                    )

                try:
                    # Execute operation for this feature
                    rows_affected = op.execute_for_feature(
                        store,
                        feature_key_str,
                        snapshot_version=migration.to_snapshot_version,
                        from_snapshot_version=migration.from_snapshot_version,
                        dry_run=dry_run,
                    )

                    # Log feature completed
                    if not dry_run:
                        self.storage.write_event(
                            Event.feature_completed(
                                project=project,
                                migration_id=migration.migration_id,
                                feature_key=feature_key_str,
                                rows_affected=rows_affected,
                            )
                        )

                    affected_features_list.append(feature_key_str)
                    rows_affected_total += rows_affected

                except Exception as e:
                    # Capture both human-friendly and full traceback messages
                    short_error = str(e) if str(e) else repr(e)
                    trace_error = traceback.format_exc().strip()
                    errors[feature_key_str] = f"{short_error}\n{trace_error}"

                    # Log exception with full traceback
                    logger.exception(f"Error in feature {feature_key_str}")

                    # Log feature failed with concise error in system tables
                    if not dry_run:
                        self.storage.write_event(
                            Event.feature_failed(
                                project=project,
                                migration_id=migration.migration_id,
                                feature_key=feature_key_str,
                                error_message=short_error,
                            )
                        )

                    continue
        else:
            # Future: Support other operation types here
            raise NotImplementedError(
                "Only DataVersionReconciliation is currently supported"
            )

        # Determine status
        if dry_run:
            status = "skipped"
        elif len(errors) == 0:
            status = "completed"
            if not dry_run:
                self.storage.write_event(
                    Event.migration_completed(
                        project=project, migration_id=migration.migration_id
                    )
                )
        else:
            status = "failed"
            if not dry_run:
                self.storage.write_event(
                    Event.migration_failed(
                        project=project,
                        migration_id=migration.migration_id,
                        error_message="",
                    )
                )

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        from metaxy.migrations.models import MigrationResult

        return MigrationResult(
            migration_id=migration.migration_id,
            status=status,
            features_completed=len(affected_features_list),
            features_failed=len(errors),
            features_skipped=len(skipped),
            affected_features=affected_features_list,
            errors={**errors, **skipped},  # Combine for display
            rows_affected=rows_affected_total,
            duration_seconds=duration,
            timestamp=start_time,
        )

    def _execute_full_graph_migration(
        self,
        migration: "FullGraphMigration",
        store: "MetadataStore",
        project: str,
        dry_run: bool,
    ) -> "MigrationResult":
        """Execute FullGraphMigration with topological sorting.

        Args:
            migration: FullGraphMigration to execute
            store: Metadata store
            project: Project name for event tracking
            dry_run: If True, only validate

        Returns:
            MigrationResult
        """
        import logging
        from datetime import datetime, timezone

        from metaxy.metadata_store.system import Event
        from metaxy.models.feature import FeatureGraph
        from metaxy.models.types import FeatureKey

        logger = logging.getLogger(__name__)

        start_time = datetime.now(timezone.utc)

        # Log migration started
        if not dry_run:
            self.storage.write_event(
                Event.migration_started(
                    project=project, migration_id=migration.migration_id
                )
            )

        affected_features_list = []
        errors = {}
        skipped = {}  # Track skipped features separately
        rows_affected_total = 0

        # Get graph for topological sorting
        graph = FeatureGraph.get_active()

        # Execute each operation (already instantiated by migration.operations property)
        # Zip with ops to maintain correspondence between operation instance and its config
        for operation, op_dict in zip(migration.operations, migration.ops):
            op_config = OperationConfig.model_validate(op_dict)

            # Sort features topologically
            feature_keys = [FeatureKey(fk.split("/")) for fk in op_config.features]
            sorted_features = graph.topological_sort_features(feature_keys)

            # Execute for each feature in topological order
            for feature_key_obj in sorted_features:
                feature_key_str = feature_key_obj.to_string()

                # Check if already completed (resume support)
                if not dry_run and self.storage.is_feature_completed(
                    migration.migration_id, feature_key_str, project
                ):
                    affected_features_list.append(feature_key_str)
                    continue

                # Check if any upstream dependencies failed in this migration run
                plan = graph.get_feature_plan(feature_key_obj)

                if plan.deps:
                    failed_deps = [
                        dep.key.to_string()
                        for dep in plan.deps
                        if dep.key.to_string() in errors
                        or dep.key.to_string() in skipped
                    ]

                    if failed_deps:
                        # Find root causes (features with actual errors, not just skipped)
                        root_causes = self._find_root_causes(
                            failed_deps, {**errors, **skipped}
                        )
                        error_msg = f"Skipped due to failed dependencies: {', '.join(root_causes)}"
                        skipped[feature_key_str] = error_msg

                        # Log as failed
                        if not dry_run:
                            self.storage.write_event(
                                Event.feature_failed(
                                    project=project,
                                    migration_id=migration.migration_id,
                                    feature_key=feature_key_str,
                                    error_message=error_msg,
                                )
                            )
                        continue

                # Log feature started
                if not dry_run:
                    self.storage.write_event(
                        Event.feature_started(
                            project=project,
                            migration_id=migration.migration_id,
                            feature_key=feature_key_str,
                        )
                    )

                try:
                    # Execute operation for this feature
                    rows_affected = operation.execute_for_feature(
                        store,
                        feature_key_str,
                        snapshot_version=migration.snapshot_version,
                        from_snapshot_version=migration.from_snapshot_version,
                        dry_run=dry_run,
                    )

                    # Log feature completed
                    if not dry_run:
                        self.storage.write_event(
                            Event.feature_completed(
                                project=project,
                                migration_id=migration.migration_id,
                                feature_key=feature_key_str,
                                rows_affected=rows_affected,
                            )
                        )

                    affected_features_list.append(feature_key_str)
                    rows_affected_total += rows_affected

                except Exception as e:
                    short_error = str(e) if str(e) else repr(e)
                    trace_error = traceback.format_exc().strip()
                    errors[feature_key_str] = f"{short_error}\n{trace_error}"

                    # Log exception with full traceback
                    logger.exception(f"Error in feature {feature_key_str}")

                    # Log feature failed
                    if not dry_run:
                        self.storage.write_event(
                            Event.feature_failed(
                                project=project,
                                migration_id=migration.migration_id,
                                feature_key=feature_key_str,
                                error_message=short_error,
                            )
                        )

                    continue

        # Determine status
        if dry_run:
            status = "skipped"
        elif len(errors) == 0:
            status = "completed"
            if not dry_run:
                self.storage.write_event(
                    Event.migration_completed(
                        project=project, migration_id=migration.migration_id
                    )
                )
        else:
            status = "failed"
            if not dry_run:
                self.storage.write_event(
                    Event.migration_failed(
                        project=project,
                        migration_id=migration.migration_id,
                        error_message="",
                    )
                )

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        from metaxy.migrations.models import MigrationResult

        return MigrationResult(
            migration_id=migration.migration_id,
            status=status,
            features_completed=len(affected_features_list),
            features_failed=len(errors),
            features_skipped=len(skipped),
            affected_features=affected_features_list,
            errors={**errors, **skipped},  # Combine for display
            rows_affected=rows_affected_total,
            duration_seconds=duration,
            timestamp=start_time,
        )
