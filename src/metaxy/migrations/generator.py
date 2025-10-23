"""Migration generation."""

from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

from metaxy.metadata_store.exceptions import FeatureNotFoundError
from metaxy.migrations.detector import detect_feature_changes
from metaxy.migrations.models import Migration
from metaxy.migrations.ops import DataVersionReconciliation
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.feature import FeatureGraph


def _is_upstream_of(
    upstream_key: FeatureKey, downstream_key: FeatureKey, graph: "FeatureGraph"
) -> bool:
    """Check if upstream_key is in the dependency chain of downstream_key.

    Args:
        upstream_key: Potential upstream feature
        downstream_key: Feature to check dependencies for
        graph: Feature graph

    Returns:
        True if upstream_key is a direct or transitive dependency of downstream_key
    """
    plan = graph.get_feature_plan(downstream_key)

    if plan.deps is None:
        return False

    # Check direct dependencies
    for dep in plan.deps:
        if dep.key == upstream_key:
            return True

    # Check transitive dependencies (recursive)
    for dep in plan.deps:
        if _is_upstream_of(upstream_key, dep.key, graph):
            return True

    return False


def generate_migration(
    store: "MetadataStore",
    *,
    from_snapshot_id: str | None = None,
    to_snapshot_id: str | None = None,
    class_path_overrides: dict[str, str] | None = None,
) -> Migration | None:
    """Generate migration from detected feature changes or between snapshots.

    Two modes of operation:

    1. **Default mode** (both snapshot_ids None):
       - Compares latest recorded snapshot (store) vs current active graph (code)
       - This is the normal workflow: detect code changes

    2. **Historical mode** (both snapshot_ids provided):
       - Reconstructs from_graph from from_snapshot_id
       - Reconstructs to_graph from to_snapshot_id
       - Compares these two historical registries
       - Useful for: backfilling migrations, testing, recovery

    Generates explicit operations for ALL affected features (root + downstream).
    Each downstream feature gets its own DataVersionReconciliation operation.

    Args:
        store: Metadata store to check
        from_snapshot_id: Optional snapshot ID to compare from (historical mode)
        to_snapshot_id: Optional snapshot ID to compare to (historical mode)
        class_path_overrides: Optional overrides for moved/renamed feature classes

    Returns:
        Migration object, or None if no changes detected

    Raises:
        ValueError: If only one snapshot_id is provided, or snapshots not found

    Example (default mode):
        >>> migration = generate_migration(store)
        >>> if migration:
        >>>     migration.to_yaml("migrations/001_update.yaml")

    Example (historical mode):
        >>> migration = generate_migration(
        ...     store,
        ...     from_snapshot_id="abc123...",
        ...     to_snapshot_id="def456...",
        ... )
    """
    from metaxy.models.feature import FeatureGraph

    # Step 1: Determine from_graph and from_snapshot_id
    # If not provided, get latest snapshot from store
    from_graph = None

    if from_snapshot_id is None:
        # Default mode: get from store's latest snapshot
        from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

        try:
            feature_versions = store.read_metadata(
                FEATURE_VERSIONS_KEY, current_only=False
            )
            if len(feature_versions) > 0:
                # Get most recent snapshot
                latest_snapshot = feature_versions.sort(
                    "recorded_at", descending=True
                ).head(1)
                from_snapshot_id = latest_snapshot["snapshot_id"][0]
                print(f"From: latest snapshot {from_snapshot_id[:16]}...")
            else:
                raise ValueError(
                    "No feature graph snapshot found in metadata store. "
                    "Run 'metaxy push' first to record feature versions before generating migrations."
                )
        except FeatureNotFoundError:
            raise ValueError(
                "No feature versions recorded yet. "
                "Run 'metaxy push' first to record the feature graph snapshot."
            )
    else:
        print(f"From: snapshot {from_snapshot_id[:16]}...")

    # Load from_graph (always from snapshot)
    from metaxy.migrations.executor import _load_historical_graph

    from_graph = _load_historical_graph(store, from_snapshot_id, class_path_overrides)

    # Step 2: Determine to_graph and to_snapshot_id
    if to_snapshot_id is None:
        # Default mode: record current active graph and use its snapshot
        # This ensures the to_snapshot is available when executing the migration
        to_snapshot_id = store.serialize_feature_graph()
        to_graph = FeatureGraph.get_active()
        print(f"To: current active graph (snapshot {to_snapshot_id[:16]}...)")
    else:
        # Historical mode: load from snapshot
        to_graph = _load_historical_graph(store, to_snapshot_id, class_path_overrides)
        print(f"To: snapshot {to_snapshot_id[:16]}...")

    # Step 3: Detect changes by comparing the two graphs
    root_operations = detect_feature_changes(
        store,
        from_graph,
        to_graph,
    )

    if not root_operations:
        print("No feature changes detected. All features up to date!")
        return None

    # Generate migration ID and timestamp
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    migration_id = f"migration_{timestamp_str}"

    # Show detected root changes
    print(f"\nDetected {len(root_operations)} root feature change(s):")
    for op in root_operations:
        feature_key_str = FeatureKey(op.feature_key).to_string()
        print(f"  ✓ {feature_key_str}")

    # Discover downstream features that need reconciliation (use to_graph)
    root_keys = [FeatureKey(op.feature_key) for op in root_operations]
    downstream_keys = to_graph.get_downstream_features(root_keys)

    # Create explicit operations for downstream features
    downstream_operations = []

    if downstream_keys:
        print(
            f"\nGenerating explicit operations for {len(downstream_keys)} downstream feature(s):"
        )

    for downstream_key in downstream_keys:
        feature_key_str = downstream_key.to_string()
        feature_cls = to_graph.features_by_key[downstream_key]

        # Check if feature exists in from_snapshot (if not, it's new - skip)
        try:
            from_metadata = store.read_metadata(
                feature_cls,
                current_only=False,
                allow_fallback=False,
                filters=pl.col("snapshot_id") == from_snapshot_id,
            )
            if len(from_metadata) == 0:
                # Feature doesn't exist in from_snapshot - it's new, skip
                print(f"  ⊘ {feature_key_str} (new feature, skipping)")
                continue
        except FeatureNotFoundError:
            # Feature not materialized yet
            print(f"  ⊘ {feature_key_str} (not materialized yet, skipping)")
            continue

        # Determine which root changes affect this downstream feature
        to_graph.get_feature_plan(downstream_key)
        affected_by = []

        for root_op in root_operations:
            root_key = FeatureKey(root_op.feature_key)
            # Check if this root is in the upstream dependency chain
            if _is_upstream_of(root_key, downstream_key, to_graph):
                affected_by.append(root_key.to_string())

        # Build informative reason
        if len(affected_by) == 1:
            reason = f"Reconcile data_versions due to changes in: {affected_by[0]}"
        else:
            reason = (
                f"Reconcile data_versions due to changes in: {', '.join(affected_by)}"
            )

        # Create operation (feature versions derived from snapshots)
        feature_key_slug = feature_key_str.replace("/", "_")
        op_id = f"reconcile_{feature_key_slug}"

        downstream_operations.append(
            DataVersionReconciliation(
                id=op_id,
                feature_key=list(downstream_key),
                reason=reason,
            )
        )

        print(f"  ✓ {feature_key_str}")

    # Combine all operations
    all_operations = root_operations + downstream_operations

    print(
        f"\nGenerated {len(all_operations)} total operations "
        f"({len(root_operations)} root + {len(downstream_operations)} downstream)"
    )

    # Find the latest migration to set as parent
    from metaxy.migrations.executor import MIGRATIONS_KEY

    parent_migration_id = None
    try:
        existing_migrations = store.read_metadata(MIGRATIONS_KEY, current_only=False)
        if len(existing_migrations) > 0:
            # Get most recent migration by created_at
            latest = existing_migrations.sort("created_at", descending=True).head(1)
            parent_migration_id = latest["migration_id"][0]
    except FeatureNotFoundError:
        # No migrations yet
        pass

    # Note: from_snapshot_id and to_snapshot_id were already resolved earlier

    # Create migration (serialize operations to dicts)
    root_count = len(root_operations)
    migration = Migration(
        version=1,
        id=migration_id,
        parent_migration_id=parent_migration_id,
        from_snapshot_id=from_snapshot_id,
        to_snapshot_id=to_snapshot_id,
        description=f"Auto-generated migration for {root_count} changed feature(s) + {len(downstream_operations)} downstream",
        created_at=timestamp,
        operations=[op.model_dump(by_alias=True) for op in all_operations],
    )

    return migration
