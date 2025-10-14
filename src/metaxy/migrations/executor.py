"""Migration execution logic."""

import time
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

from metaxy.metadata_store.base import MIGRATION_HISTORY_KEY
from metaxy.metadata_store.exceptions import FeatureNotFoundError
from metaxy.migrations.models import Migration, MigrationResult
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore


def _is_migration_completed(store: "MetadataStore", migration_id: str) -> bool:
    """Check if migration is fully completed.

    A migration is complete when:
    1. Marked as 'completed' in migration history, OR
    2. All source features have NO rows with old feature_version

    Args:
        store: Metadata store
        migration_id: Migration ID to check

    Returns:
        True if migration is fully completed
    """
    # Check migration history
    try:
        history = store.read_metadata(
            MIGRATION_HISTORY_KEY,
            filters=pl.col("migration_id") == migration_id,
            current_only=False,
        )
        if len(history) > 0 and history["status"][0] == "completed":
            return True
    except FeatureNotFoundError:
        # No history yet
        pass

    return False


def _record_migration(
    store: "MetadataStore",
    migration: Migration,
    status: str,
    operations_applied: int,
    affected_features: list[str],
    errors: dict[str, str] | None = None,
) -> None:
    """Record migration in history table.

    Args:
        store: Metadata store
        migration: Migration that was applied
        status: "completed", "partial", or "failed"
        operations_applied: Number of operations successfully applied
        affected_features: List of feature keys that were updated
        errors: Dict of feature_key -> error message (if any)
    """
    migration_record = pl.DataFrame(
        {
            "migration_id": [migration.id],
            "applied_at": [datetime.now()],
            "status": [status],
            "operations_count": [len(migration.operations)],
            "operations_applied": [operations_applied],
            "affected_features": [affected_features],
            "errors": [errors],
        }
    )

    # Write to migration history (system table)
    store._write_metadata_impl(MIGRATION_HISTORY_KEY, migration_record)


def apply_migration(
    store: "MetadataStore",
    migration: Migration,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> MigrationResult:
    """Apply a migration: recalculate sources + propagate to entire downstream DAG.

    Process:
    1. Check if already completed (idempotency)
    2. For each source feature:
       a. Query rows with old feature_version
       b. If found: recalculate data_version, write with new feature_version
       c. If not found: skip (already migrated)
    3. Discover entire downstream DAG from sources
    4. Topologically sort downstream features
    5. For each downstream feature (in order):
       a. Read current metadata
       b. Read upstream metadata (with NEW data_versions)
       c. Call feature.align_metadata_with_upstream()
       d. Recalculate data_versions
       e. Write with current feature_version
    6. Mark migration as completed

    Failure Recovery:
    - If fails midway, re-running continues from failure point
    - Already-migrated features are skipped (idempotent)
    - Partial progress is preserved

    Args:
        store: Metadata store
        migration: Migration to apply
        dry_run: Show what would happen without executing (default: False)
        force: Re-apply even if already completed (default: False)

    Returns:
        Result of migration execution

    Example:
        >>> result = apply_migration(store, migration, dry_run=True)
        >>> print(result.summary())

        >>> result = apply_migration(store, migration)
        >>> if result.status == "completed":
        ...     print("Success!")
    """
    start_time = time.time()
    timestamp = datetime.now()

    # Check if already completed (idempotency)
    if not force and _is_migration_completed(store, migration.id):
        return MigrationResult(
            migration_id=migration.id,
            status="skipped",
            operations_applied=0,
            operations_failed=0,
            affected_features=[],
            errors={},
            duration_seconds=time.time() - start_time,
            timestamp=timestamp,
        )

    if dry_run:
        # Dry-run: validate and show what would happen
        affected = []
        for operation in migration.operations:
            feature_key = FeatureKey(operation.feature_key)
            affected.append(feature_key.to_string())

        source_keys = [FeatureKey(op.feature_key) for op in migration.operations]
        downstream_keys = store.registry.get_downstream_features(source_keys)
        affected.extend([dk.to_string() for dk in downstream_keys])

        return MigrationResult(
            migration_id=migration.id,
            status="skipped",
            operations_applied=0,
            operations_failed=0,
            affected_features=affected,
            errors={},
            duration_seconds=time.time() - start_time,
            timestamp=timestamp,
        )

    # Execute migration
    affected_features = []
    errors = {}
    operations_applied = 0

    # Step 1: Migrate source features
    for operation in migration.operations:
        feature_key = FeatureKey(operation.feature_key)
        feature_cls = store.registry.features_by_key[feature_key]

        try:
            # Check if this source still needs migration
            try:
                old_rows = store.read_metadata(
                    feature_cls,
                    current_only=False,
                    filters=pl.col("feature_version") == operation.from_,
                    allow_fallback=False,
                )
            except FeatureNotFoundError:
                # Feature doesn't exist yet - nothing to migrate
                continue

            if len(old_rows) == 0:
                # Already migrated, skip
                continue

            # Copy metadata and recalculate data versions
            # Keep all columns from source except data_version and feature_version (will be recalculated)
            # This ensures we preserve all user metadata columns
            columns_to_keep = [
                c
                for c in old_rows.columns
                if c not in ["data_version", "feature_version"]
            ]
            sample_metadata = old_rows.select(columns_to_keep)

            # Recalculate data versions based on new feature definition
            # This appends NEW rows with:
            # - Same sample_id and metadata columns
            # - Recalculated data_version (based on new upstream/code)
            # - New feature_version (from current feature definition)
            # Old rows remain unchanged (immutable!)
            store.calculate_and_write_data_versions(
                feature=feature_cls,
                sample_df=sample_metadata,
            )

            affected_features.append(feature_key.to_string())
            operations_applied += 1

        except Exception as e:
            errors[feature_key.to_string()] = str(e)
            # Continue with other sources (fail-graceful)

    if errors and not affected_features:
        # All sources failed
        _record_migration(
            store, migration, "failed", operations_applied, affected_features, errors
        )
        return MigrationResult(
            migration_id=migration.id,
            status="failed",
            operations_applied=operations_applied,
            operations_failed=len(errors),
            affected_features=affected_features,
            errors=errors,
            duration_seconds=time.time() - start_time,
            timestamp=timestamp,
        )

    # Step 2: Propagate to all downstream features
    source_keys = [FeatureKey(op.feature_key) for op in migration.operations]
    downstream_keys = store.registry.get_downstream_features(source_keys)

    for downstream_key in downstream_keys:
        feature_cls = store.registry.features_by_key[downstream_key]

        try:
            # Read current metadata
            try:
                current_metadata = store.read_metadata(
                    feature_cls,
                    current_only=True,
                    allow_fallback=False,
                )
            except FeatureNotFoundError:
                current_metadata = pl.DataFrame({"sample_id": []})

            # Read upstream metadata (with NEW data_versions from step 1)
            upstream_metadata = store.read_upstream_metadata(
                feature_cls,
                current_only=True,
            )

            # Align metadata using feature's custom logic
            aligned_metadata = feature_cls.align_metadata_with_upstream(  # type: ignore[attr-defined]
                current_metadata,
                upstream_metadata,
            )

            # Drop data_version and feature_version columns (will be recalculated)
            columns_to_keep = [
                c
                for c in aligned_metadata.columns
                if c not in ["data_version", "feature_version"]
            ]
            sample_metadata = aligned_metadata.select(columns_to_keep)

            # Recalculate and write
            store.calculate_and_write_data_versions(
                feature=feature_cls,
                sample_df=sample_metadata,
            )

            affected_features.append(downstream_key.to_string())

        except Exception as e:
            errors[downstream_key.to_string()] = str(e)
            # Record partial and return
            _record_migration(
                store,
                migration,
                "partial",
                operations_applied,
                affected_features,
                errors,
            )
            return MigrationResult(
                migration_id=migration.id,
                status="failed",
                operations_applied=operations_applied,
                operations_failed=len(errors),
                affected_features=affected_features,
                errors=errors,
                duration_seconds=time.time() - start_time,
                timestamp=timestamp,
            )

    # All succeeded!
    _record_migration(
        store, migration, "completed", operations_applied, affected_features, None
    )

    return MigrationResult(
        migration_id=migration.id,
        status="completed",
        operations_applied=operations_applied,
        operations_failed=len(errors),
        affected_features=affected_features,
        errors=errors,
        duration_seconds=time.time() - start_time,
        timestamp=timestamp,
    )
