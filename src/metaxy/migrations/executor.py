"""Migration execution logic with 3-table progress tracking."""

import time
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

from metaxy.metadata_store.exceptions import FeatureNotFoundError
from metaxy.migrations.models import Migration, MigrationResult
from metaxy.migrations.ops import BaseOperation
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.feature import FeatureRegistry

# System table keys for migration tracking
MIGRATIONS_KEY = FeatureKey(["__metaxy__", "migrations"])
MIGRATION_OPS_KEY = FeatureKey(["__metaxy__", "migration_ops"])
MIGRATION_OP_STEPS_KEY = FeatureKey(["__metaxy__", "migration_op_steps"])


class MigrationStatus:
    """Query and check migration completion status.

    Provides methods to check completion at migration, operation, and step levels.
    All status is derived from immutable system tables, not stored.
    """

    def __init__(self, store: "MetadataStore"):
        self.store = store

    def is_step_complete(
        self, migration_id: str, operation_id: str, feature_key: str
    ) -> bool:
        """Check if specific step completed successfully.

        Args:
            migration_id: Migration ID
            operation_id: Operation ID
            feature_key: Feature key (string)

        Returns:
            True if step completed without errors
        """
        try:
            steps = self.store.read_metadata(
                MIGRATION_OP_STEPS_KEY,
                filters=(
                    (pl.col("migration_id") == migration_id)
                    & (pl.col("operation_id") == operation_id)
                    & (pl.col("feature_key") == feature_key)
                    & (pl.col("error").is_null())
                ),
                current_only=False,
            )
            return len(steps) > 0
        except FeatureNotFoundError:
            return False

    def is_operation_complete(self, migration_id: str, operation_id: str) -> bool:
        """Check if operation is complete.

        An operation is complete when all expected_steps have successful completion records.

        Args:
            migration_id: Migration ID
            operation_id: Operation ID

        Returns:
            True if all steps completed successfully
        """
        # Get operation definition
        try:
            op_def = self.store.read_metadata(
                MIGRATION_OPS_KEY,
                filters=(
                    (pl.col("migration_id") == migration_id)
                    & (pl.col("operation_id") == operation_id)
                ),
                current_only=False,
            )
        except FeatureNotFoundError:
            return False

        if len(op_def) == 0:
            return False

        expected_steps_raw = op_def["expected_steps"][0]

        # Convert to Python list (handles JSON string, Polars Series, or native list)
        if isinstance(expected_steps_raw, str):
            import json

            expected_steps = json.loads(expected_steps_raw)
        elif hasattr(expected_steps_raw, "to_list"):
            # Polars Series (from SQLite/DuckDB deserialization)
            expected_steps = expected_steps_raw.to_list()
        else:
            # Already a list
            expected_steps = expected_steps_raw

        # Check if all steps completed
        for feature_key in expected_steps:
            if not self.is_step_complete(migration_id, operation_id, feature_key):
                return False

        return True

    def is_migration_complete(self, migration_id: str) -> bool:
        """Check if migration is complete.

        A migration is complete when all operations are complete.

        Args:
            migration_id: Migration ID

        Returns:
            True if all operations completed successfully
        """
        # Get migration definition
        try:
            migration = self.store.read_metadata(
                MIGRATIONS_KEY,
                filters=pl.col("migration_id") == migration_id,
                current_only=False,
            )
        except FeatureNotFoundError:
            return False

        if len(migration) == 0:
            return False

        expected_op_ids_raw = migration["operation_ids"][0]

        # Convert to Python list (handles JSON string, Polars Series, or native list)
        if isinstance(expected_op_ids_raw, str):
            import json

            expected_op_ids = json.loads(expected_op_ids_raw)
        elif hasattr(expected_op_ids_raw, "to_list"):
            # Polars Series (from SQLite/DuckDB deserialization)
            expected_op_ids = expected_op_ids_raw.to_list()
        else:
            # Already a list
            expected_op_ids = expected_op_ids_raw

        # Check if all operations completed
        for op_id in expected_op_ids:
            if not self.is_operation_complete(migration_id, op_id):
                return False

        return True


# ========== Helper Functions for 3-Table System ==========


def _load_historical_registry(
    store: "MetadataStore",
    snapshot_id: str,
    class_path_overrides: dict[str, str] | None = None,
) -> "FeatureRegistry":
    """Load historical registry from snapshot.

    Args:
        store: Metadata store
        snapshot_id: Snapshot ID to reconstruct
        class_path_overrides: Optional overrides for moved/renamed feature classes

    Returns:
        FeatureRegistry reconstructed from snapshot

    Raises:
        ValueError: If snapshot not found
        ImportError: If feature class cannot be imported
    """
    from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY
    from metaxy.metadata_store.exceptions import FeatureNotFoundError
    from metaxy.models.feature import FeatureRegistry

    # Load all features from this snapshot
    try:
        features_data = store.read_metadata(
            FEATURE_VERSIONS_KEY,
            filters=pl.col("snapshot_id") == snapshot_id,
            current_only=False,
        )
    except FeatureNotFoundError:
        raise ValueError(
            f"Snapshot '{snapshot_id}' not found in metadata store. "
            f"Cannot reconstruct historical feature graph."
        )

    if len(features_data) == 0:
        raise ValueError(
            f"Snapshot '{snapshot_id}' is empty. "
            f"No features recorded with this snapshot ID."
        )

    # Build snapshot dict for from_snapshot()
    snapshot_dict = {}
    for row in features_data.iter_rows(named=True):
        feature_key_str = row["feature_key"]
        feature_spec_raw = row["feature_spec"]
        feature_class_path = row.get("feature_class_path")

        # Parse feature_spec (handle both JSON string and struct)
        if isinstance(feature_spec_raw, str):
            import json

            feature_spec_dict = json.loads(feature_spec_raw)
        else:
            # Already parsed (DuckDB/ClickHouse return structs)
            feature_spec_dict = feature_spec_raw

        snapshot_dict[feature_key_str] = {
            "feature_spec": feature_spec_dict,
            "feature_class_path": feature_class_path,
        }

    # Reconstruct registry from snapshot (with optional overrides)
    return FeatureRegistry.from_snapshot(
        snapshot_dict, class_path_overrides=class_path_overrides
    )


def _is_migration_registered(store: "MetadataStore", migration_id: str) -> bool:
    """Check if migration is registered in migrations table."""
    try:
        migrations = store.read_metadata(
            MIGRATIONS_KEY,
            filters=pl.col("migration_id") == migration_id,
            current_only=False,
        )
        return len(migrations) > 0
    except FeatureNotFoundError:
        return False


def _register_migration(store: "MetadataStore", migration: Migration) -> None:
    """Record migration declaration in migrations table (idempotent).

    Args:
        store: Metadata store
        migration: Migration to register
    """
    import json

    if _is_migration_registered(store, migration.id):
        return  # Already registered

    # Serialize complex types to JSON strings for storage compatibility
    operation_ids_json = json.dumps([op.get("id") for op in migration.operations])
    migration_yaml_json = json.dumps(migration.model_dump(mode="json", by_alias=True))

    migration_record = pl.DataFrame(
        {
            "migration_id": [migration.id],
            "created_at": [migration.created_at],
            "description": [migration.description],
            "operation_ids": [operation_ids_json],
            "migration_yaml": [migration_yaml_json],
        },
        schema={
            "migration_id": pl.String,
            "created_at": pl.Datetime("us"),
            "description": pl.String,
            "operation_ids": pl.String,
            "migration_yaml": pl.String,
        },
    )

    store._write_metadata_impl(MIGRATIONS_KEY, migration_record)


def _is_operation_registered(
    store: "MetadataStore", migration_id: str, operation_id: str
) -> bool:
    """Check if operation is registered in ops table."""
    try:
        ops = store.read_metadata(
            MIGRATION_OPS_KEY,
            filters=(
                (pl.col("migration_id") == migration_id)
                & (pl.col("operation_id") == operation_id)
            ),
            current_only=False,
        )
        return len(ops) > 0
    except FeatureNotFoundError:
        return False


def _register_operation(
    store: "MetadataStore", migration_id: str, operation: BaseOperation
) -> None:
    """Record operation definition in ops table (idempotent).

    Computes expected_steps (root feature only - no implicit downstream).

    Args:
        store: Metadata store
        migration_id: Parent migration ID
        operation: Operation to register
    """
    import json

    if _is_operation_registered(store, migration_id, operation.id):
        return  # Already registered

    # Expected steps is just the root feature (no implicit downstream)
    # All features are explicit operations in the migration YAML
    expected_steps_list = [FeatureKey(operation.feature_key).to_string()]
    expected_steps_json = json.dumps(expected_steps_list)

    op_record = pl.DataFrame(
        {
            "migration_id": [migration_id],
            "operation_id": [operation.id],
            "operation_type": [operation.type],  # type: ignore[attr-defined]
            "feature_key": [FeatureKey(operation.feature_key).to_string()],
            "expected_steps": [expected_steps_json],
            "operation_config_hash": [operation.operation_config_hash()],
            "created_at": [datetime.now()],
        },
        schema={
            "migration_id": pl.String,
            "operation_id": pl.String,
            "operation_type": pl.String,
            "feature_key": pl.String,
            "expected_steps": pl.String,
            "operation_config_hash": pl.String,
            "created_at": pl.Datetime("us"),
        },
    )

    store._write_metadata_impl(MIGRATION_OPS_KEY, op_record)


def _validate_operation_not_changed(
    store: "MetadataStore", migration_id: str, operation: BaseOperation
) -> None:
    """Validate operation config hasn't changed since registration.

    Args:
        store: Metadata store
        migration_id: Migration ID
        operation: Current operation from YAML

    Raises:
        ValueError: If operation content changed
    """
    if not _is_operation_registered(store, migration_id, operation.id):
        return  # Not registered yet, nothing to validate

    stored_op = store.read_metadata(
        MIGRATION_OPS_KEY,
        filters=(
            (pl.col("migration_id") == migration_id)
            & (pl.col("operation_id") == operation.id)
        ),
        current_only=False,
    )

    if len(stored_op) == 0:
        return

    stored_hash = stored_op["operation_config_hash"][0]
    current_hash = operation.operation_config_hash()

    if stored_hash != current_hash:
        raise ValueError(
            f"Operation '{operation.id}' content changed since last run. "
            f"This is unsafe after partial migration. "
            f"Revert changes or use --force to restart migration."
        )


def _is_step_complete(
    store: "MetadataStore", migration_id: str, operation_id: str, feature_key: str
) -> bool:
    """Check if specific step already completed successfully.

    Delegates to MigrationStatus class.
    """
    return MigrationStatus(store).is_step_complete(
        migration_id, operation_id, feature_key
    )


def _record_step_completion(
    store: "MetadataStore",
    migration_id: str,
    operation_id: str,
    feature_key: str,
    step_type: str,
    rows_affected: int,
    error: str | None,
) -> None:
    """Record feature step completion/failure in steps table.

    Args:
        store: Metadata store
        migration_id: Migration ID
        operation_id: Operation ID
        feature_key: Feature key (string)
        step_type: "operation_execution"
        rows_affected: Number of rows processed
        error: Error message if failed, None if successful
    """
    step_record = pl.DataFrame(
        {
            "migration_id": [migration_id],
            "operation_id": [operation_id],
            "feature_key": [feature_key],
            "step_type": [step_type],
            "completed_at": [datetime.now()],
            "rows_affected": [rows_affected],
            "error": [error],
        },
        schema={
            "migration_id": pl.String,
            "operation_id": pl.String,
            "feature_key": pl.String,
            "step_type": pl.String,
            "completed_at": pl.Datetime("us"),
            "rows_affected": pl.Int64,
            "error": pl.String,  # Always String, even when None
        },
    )

    store._write_metadata_impl(MIGRATION_OP_STEPS_KEY, step_record)


def _is_operation_complete(
    store: "MetadataStore", migration_id: str, operation_id: str
) -> bool:
    """Check if operation is complete.

    Delegates to MigrationStatus class.
    """
    return MigrationStatus(store).is_operation_complete(migration_id, operation_id)


def _is_migration_complete(store: "MetadataStore", migration_id: str) -> bool:
    """Check if migration is complete.

    Delegates to MigrationStatus class.
    """
    return MigrationStatus(store).is_migration_complete(migration_id)


# ========== Main Execution Function ==========


def apply_migration(
    store: "MetadataStore",
    migration: Migration,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> MigrationResult:
    """Apply a migration with fine-grained progress tracking.

    Process:
    1. Validate parent migration is completed (if specified)
    2. Register migration and operations in system tables (idempotent)
    3. For each operation:
       a. Validate operation hasn't changed (content hash check)
       b. Execute operation (calls operation.execute())
       c. Record step completion
    4. Derive final status from system tables

    All operations are explicit in YAML - no implicit downstream propagation.
    The migration generator creates explicit operations for all affected features.

    Progress is tracked at three levels:
    - Migration level (migrations table)
    - Operation level (migration_ops table)
    - Feature step level (migration_op_steps table)

    All tables are immutable/append-only. Status is derived at query time.

    Migrations form an explicit dependency chain via parent_migration_id.
    Parent migrations must be completed before applying child migrations.

    Args:
        store: Metadata store
        migration: Migration to apply
        dry_run: Show what would happen without executing (default: False)
        force: Re-apply even if already completed, skip validation (default: False)

    Returns:
        Result of migration execution

    Raises:
        ValueError: If parent migration is not completed

    Example:
        >>> result = apply_migration(store, migration, dry_run=True)
        >>> print(result.summary())

        >>> result = apply_migration(store, migration)
        >>> if result.status == "completed":
        ...     print("Success!")
    """
    start_time = time.time()
    timestamp = datetime.now()

    # Load historical registry from the "from" snapshot
    # This ensures migrations use the exact feature graph topology that existed
    # in the store when the migration was created, protecting against code refactoring
    historical_registry = _load_historical_registry(
        store,
        migration.from_snapshot_id,
        class_path_overrides=migration.feature_class_overrides,
    )

    # Execute migration within historical registry context
    with historical_registry.use():
        # Parse operations from dict to objects
        # Operations will use historical registry via FeatureRegistry.get_active()
        operations = migration.get_operations()

        # 1. Validate parent migration is completed (if specified)
        if migration.parent_migration_id is not None and not force and not dry_run:
            if not _is_migration_complete(store, migration.parent_migration_id):
                raise ValueError(
                    f"Parent migration '{migration.parent_migration_id}' must be completed "
                    f"before applying migration '{migration.id}'. "
                    f"Apply parent migration first or use --force to skip validation."
                )

        # 2. Register migration (idempotent)
        if not dry_run:
            _register_migration(store, migration)

        # 3. Check if already complete
        if not force and not dry_run and _is_migration_complete(store, migration.id):
            return MigrationResult(
                migration_id=migration.id,
                status="skipped",
                operations_applied=len(operations),
                operations_failed=0,
                affected_features=[],
                errors={},
                duration_seconds=time.time() - start_time,
                timestamp=timestamp,
            )

        # 4. Execute each operation (using historical registry)
        affected_features = []
        errors = {}
        operations_applied = 0

        for operation in operations:
            # Register operation (idempotent)
            if not dry_run:
                _register_operation(store, migration.id, operation)

                # Validate operation hasn't changed (unless force)
                if not force:
                    try:
                        _validate_operation_not_changed(store, migration.id, operation)
                    except ValueError as e:
                        errors[operation.id] = str(e)
                        continue

            feature_key_str = FeatureKey(operation.feature_key).to_string()

            # Check if step already complete (idempotent)
            if not dry_run and _is_step_complete(
                store, migration.id, operation.id, feature_key_str
            ):
                # Already completed, skip
                affected_features.append(feature_key_str)
                operations_applied += 1
                continue

            # Execute operation
            try:
                rows_affected = operation.execute(store, dry_run=dry_run)

                # Record step completion (unless dry-run)
                if not dry_run:
                    _record_step_completion(
                        store,
                        migration.id,
                        operation.id,
                        feature_key_str,
                        "operation_execution",
                        rows_affected,
                        None,  # No error
                    )

                affected_features.append(feature_key_str)
                operations_applied += 1

            except Exception as e:
                error_msg = str(e)
                errors[operation.id] = error_msg

                # Record step failure
                if not dry_run:
                    _record_step_completion(
                        store,
                        migration.id,
                        operation.id,
                        feature_key_str,
                        "operation_execution",
                        0,
                        error_msg,
                    )

                # Continue with other operations (fail-graceful)
                continue

        # 5. Determine final status (after all operations)
        if dry_run:
            status = "skipped"
        elif operations_applied == len(operations):
            status = "completed"
        elif operations_applied == 0:
            status = "failed"
        else:
            status = "failed"  # Partial execution treated as failed

        return MigrationResult(
            migration_id=migration.id,
            status=status,
            operations_applied=operations_applied,
            operations_failed=len(errors),
            affected_features=affected_features,
            errors=errors,
            duration_seconds=time.time() - start_time,
            timestamp=timestamp,
        )
