"""Migration executor using event-based tracking.

This is the new executor that replaces the old 3-table system with a single
event-based system stored in system tables via SystemTableStorage.
"""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

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


class _FeatureExecutionContext:
    """Context for tracking feature execution state during migration."""

    def __init__(self) -> None:
        self.affected_features: list[str] = []
        self.errors: dict[str, str] = {}
        self.skipped: dict[str, str] = {}
        self.rows_affected_total: int = 0


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
        """Find the root cause features (features with actual errors, not skipped)."""
        root_causes = []
        for dep in failed_deps:
            if dep not in errors:
                continue
            error_msg = errors[dep]
            if error_msg.startswith("Skipped due to failed dependencies:"):
                deps_part = error_msg.split(":", 1)[1].strip()
                transitive_deps = [d.strip() for d in deps_part.split(",")]
                root_causes.extend(self._find_root_causes(transitive_deps, errors))
            else:
                root_causes.append(dep)
        return list(dict.fromkeys(root_causes))

    def _log_migration_started(
        self, project: str, migration_id: str, dry_run: bool
    ) -> None:
        """Log migration started event."""
        if not dry_run:
            from metaxy.metadata_store.system import Event

            self.storage.write_event(
                Event.migration_started(project=project, migration_id=migration_id)
            )

    def _check_failed_dependencies(
        self,
        feature_key_str: str,
        plan: Any,
        ctx: _FeatureExecutionContext,
        migration_id: str,
        project: str,
        dry_run: bool,
    ) -> bool:
        """Check if feature should be skipped due to failed dependencies.

        Returns True if feature should be skipped.
        """
        if not plan.deps:
            return False

        all_errors = {**ctx.errors, **ctx.skipped}
        failed_deps = [
            dep.key.to_string()
            for dep in plan.deps
            if dep.key.to_string() in all_errors
        ]

        if not failed_deps:
            return False

        root_causes = self._find_root_causes(failed_deps, all_errors)
        error_msg = f"Skipped due to failed dependencies: {', '.join(root_causes)}"
        ctx.skipped[feature_key_str] = error_msg

        if not dry_run:
            from metaxy.metadata_store.system import Event

            self.storage.write_event(
                Event.feature_failed(
                    project=project,
                    migration_id=migration_id,
                    feature_key=feature_key_str,
                    error_message=error_msg,
                )
            )
        return True

    def _execute_feature_operation(
        self,
        operation: Any,
        store: "MetadataStore",
        feature_key_str: str,
        snapshot_version: str,
        from_snapshot_version: str | None,
        migration_id: str,
        project: str,
        ctx: _FeatureExecutionContext,
        dry_run: bool,
    ) -> None:
        """Execute an operation for a single feature."""
        logger = logging.getLogger(__name__)

        if not dry_run:
            from metaxy.metadata_store.system import Event

            self.storage.write_event(
                Event.feature_started(
                    project=project,
                    migration_id=migration_id,
                    feature_key=feature_key_str,
                )
            )

        try:
            rows_affected = operation.execute_for_feature(
                store,
                feature_key_str,
                snapshot_version=snapshot_version,
                from_snapshot_version=from_snapshot_version,
                dry_run=dry_run,
            )

            if not dry_run:
                from metaxy.metadata_store.system import Event

                self.storage.write_event(
                    Event.feature_completed(
                        project=project,
                        migration_id=migration_id,
                        feature_key=feature_key_str,
                        rows_affected=rows_affected,
                    )
                )

            ctx.affected_features.append(feature_key_str)
            ctx.rows_affected_total += rows_affected

        except Exception as e:
            error_msg = str(e) if str(e) else repr(e)
            ctx.errors[feature_key_str] = error_msg
            logger.exception(f"Error in feature {feature_key_str}")

            if not dry_run:
                from metaxy.metadata_store.system import Event

                self.storage.write_event(
                    Event.feature_failed(
                        project=project,
                        migration_id=migration_id,
                        feature_key=feature_key_str,
                        error_message=error_msg,
                    )
                )

    def _finalize_migration(
        self,
        migration_id: str,
        project: str,
        ctx: _FeatureExecutionContext,
        start_time: datetime,
        dry_run: bool,
    ) -> "MigrationResult":
        """Finalize migration and return result."""
        from metaxy.migrations.models import MigrationResult

        if dry_run:
            status = "skipped"
        elif len(ctx.errors) == 0:
            status = "completed"
            from metaxy.metadata_store.system import Event

            self.storage.write_event(
                Event.migration_completed(project=project, migration_id=migration_id)
            )
        else:
            status = "failed"
            from metaxy.metadata_store.system import Event

            self.storage.write_event(
                Event.migration_failed(
                    project=project,
                    migration_id=migration_id,
                    error_message="",
                )
            )

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        return MigrationResult(
            migration_id=migration_id,
            status=status,
            features_completed=len(ctx.affected_features),
            features_failed=len(ctx.errors),
            features_skipped=len(ctx.skipped),
            affected_features=ctx.affected_features,
            errors={**ctx.errors, **ctx.skipped},
            rows_affected=ctx.rows_affected_total,
            duration_seconds=duration,
            timestamp=start_time,
        )

    def execute(
        self,
        migration: "Migration",
        store: "MetadataStore",
        project: str,
        *,
        dry_run: bool = False,
        rerun: bool = False,
    ) -> "MigrationResult":
        """Execute migration with event logging and resumability.

        Process:
        1. Log migration_started event
        2. Get features to process from migration
        3. Sort features topologically using FeatureGraph.topological_sort_features()
        4. For each feature:
           - Check if already completed (resume support, unless rerun=True)
           - Log feature_started
           - Execute migration logic
           - Log feature_completed/failed
        5. Log migration_completed/failed

        Args:
            migration: Migration to execute
            store: Metadata store to operate on
            project: Project name for event tracking
            dry_run: If True, only validate without executing
            rerun: If True, re-run all steps including already completed ones

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
                migration, store, project, dry_run=dry_run, rerun=rerun
            )
        elif isinstance(migration, FullGraphMigration):
            return self._execute_full_graph_migration(
                migration, store, project, dry_run=dry_run, rerun=rerun
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
        rerun: bool = False,
    ) -> "MigrationResult":
        """Execute DiffMigration with topological sorting."""
        from metaxy.migrations.ops import DataVersionReconciliation
        from metaxy.models.feature import FeatureGraph
        from metaxy.models.types import FeatureKey

        start_time = datetime.now(timezone.utc)
        self._log_migration_started(project, migration.migration_id, dry_run)

        ctx = _FeatureExecutionContext()
        graph = FeatureGraph.get_active()

        if not (
            len(migration.operations) == 1
            and isinstance(migration.operations[0], DataVersionReconciliation)
        ):
            raise NotImplementedError(
                "Only DataVersionReconciliation is currently supported"
            )

        op = migration.operations[0]
        op_config = OperationConfig.model_validate(migration.ops[0])
        affected_features_to_process = self._get_diff_features_to_process(
            op_config, graph, migration, store, project
        )

        for feature_key_str in affected_features_to_process:
            if self._should_skip_completed_feature(
                migration.migration_id, feature_key_str, project, dry_run, rerun, ctx
            ):
                continue

            feature_key_obj = FeatureKey(feature_key_str.split("/"))
            plan = graph.get_feature_plan(feature_key_obj)

            if self._check_failed_dependencies(
                feature_key_str, plan, ctx, migration.migration_id, project, dry_run
            ):
                continue

            self._execute_feature_operation(
                op,
                store,
                feature_key_str,
                migration.to_snapshot_version,
                migration.from_snapshot_version,
                migration.migration_id,
                project,
                ctx,
                dry_run,
            )

        return self._finalize_migration(
            migration.migration_id, project, ctx, start_time, dry_run
        )

    def _get_diff_features_to_process(
        self,
        op_config: OperationConfig,
        graph: Any,
        migration: "DiffMigration",
        store: "MetadataStore",
        project: str,
    ) -> list[str]:
        """Get features to process for a diff migration."""
        from metaxy.models.types import FeatureKey

        if op_config.features:
            feature_keys = [FeatureKey(fk.split("/")) for fk in op_config.features]
            sorted_features = graph.topological_sort_features(feature_keys)
            return [fk.to_string() for fk in sorted_features]
        return migration.get_affected_features(store, project)

    def _should_skip_completed_feature(
        self,
        migration_id: str,
        feature_key_str: str,
        project: str,
        dry_run: bool,
        rerun: bool,
        ctx: _FeatureExecutionContext,
    ) -> bool:
        """Check if feature was already completed and should be skipped."""
        if dry_run or rerun:
            return False
        if self.storage.is_feature_completed(migration_id, feature_key_str, project):
            ctx.affected_features.append(feature_key_str)
            return True
        return False

    def _execute_full_graph_migration(
        self,
        migration: "FullGraphMigration",
        store: "MetadataStore",
        project: str,
        dry_run: bool,
        rerun: bool = False,
    ) -> "MigrationResult":
        """Execute FullGraphMigration with topological sorting."""
        from metaxy.models.feature import FeatureGraph
        from metaxy.models.types import FeatureKey

        start_time = datetime.now(timezone.utc)
        self._log_migration_started(project, migration.migration_id, dry_run)

        ctx = _FeatureExecutionContext()
        graph = FeatureGraph.get_active()

        for operation, op_dict in zip(migration.operations, migration.ops):
            op_config = OperationConfig.model_validate(op_dict)
            feature_keys = [FeatureKey(fk.split("/")) for fk in op_config.features]
            sorted_features = graph.topological_sort_features(feature_keys)

            for feature_key_obj in sorted_features:
                feature_key_str = feature_key_obj.to_string()

                if self._should_skip_completed_feature(
                    migration.migration_id,
                    feature_key_str,
                    project,
                    dry_run,
                    rerun,
                    ctx,
                ):
                    continue

                plan = graph.get_feature_plan(feature_key_obj)
                if self._check_failed_dependencies(
                    feature_key_str,
                    plan,
                    ctx,
                    migration.migration_id,
                    project,
                    dry_run,
                ):
                    continue

                self._execute_feature_operation(
                    operation,
                    store,
                    feature_key_str,
                    migration.snapshot_version,
                    migration.from_snapshot_version,
                    migration.migration_id,
                    project,
                    ctx,
                    dry_run,
                )

        return self._finalize_migration(
            migration.migration_id, project, ctx, start_time, dry_run
        )
