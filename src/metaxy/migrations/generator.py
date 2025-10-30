"""Migration generation."""

from datetime import datetime
from typing import TYPE_CHECKING

import narwhals as nw

from metaxy.graph.diff.differ import GraphDiffer
from metaxy.metadata_store.exceptions import FeatureNotFoundError
from metaxy.migrations.models import DiffMigration
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
    project: str,
    from_snapshot_version: str | None = None,
    to_snapshot_version: str | None = None,
    class_path_overrides: dict[str, str] | None = None,
) -> DiffMigration | None:
    """Generate migration from detected feature changes or between snapshots.

    Two modes of operation:

    1. **Default mode** (both snapshot_versions None):
       - Compares latest recorded snapshot (store) vs current active graph (code)
       - This is the normal workflow: detect code changes

    2. **Historical mode** (both snapshot_versions provided):
       - Reconstructs from_graph from from_snapshot_version
       - Reconstructs to_graph from to_snapshot_version
       - Compares these two historical registries
       - Useful for: backfilling migrations, testing, recovery

    Generates explicit operations for ALL affected features (root + downstream).
    Each downstream feature gets its own DataVersionReconciliation operation.

    Args:
        store: Metadata store to check
        project: Project name for filtering snapshots
        from_snapshot_version: Optional snapshot version to compare from (historical mode)
        to_snapshot_version: Optional snapshot version to compare to (historical mode)
        class_path_overrides: Optional overrides for moved/renamed feature classes

    Returns:
        Migration object, or None if no changes detected

    Raises:
        ValueError: If only one snapshot_version is provided, or snapshots not found

    Example (default mode):
        >>> migration = generate_migration(store, project="my_project")
        >>> if migration:
        >>>     migration.to_yaml("migrations/001_update.yaml")

    Example (historical mode):
        >>> migration = generate_migration(
        ...     store,
        ...     project="my_project",
        ...     from_snapshot_version="abc123...",
        ...     to_snapshot_version="def456...",
        ... )
    """
    from metaxy.models.feature import FeatureGraph

    if from_snapshot_version is None:
        # Default mode: get from store's latest snapshot
        from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

        try:
            feature_versions = store.read_metadata(
                FEATURE_VERSIONS_KEY, current_only=False
            )
            # Get most recent snapshot - only collect the top row
            latest_snapshot = nw.from_native(
                feature_versions.sort("recorded_at", descending=True).head(1).collect()
            )
            if latest_snapshot.shape[0] > 0:
                from_snapshot_version = latest_snapshot["snapshot_version"][0]
                print(f"From: latest snapshot {from_snapshot_version}...")
            else:
                raise ValueError(
                    "No feature graph snapshot found in metadata store. "
                    "Run 'metaxy graph push' first to record feature versions before generating migrations."
                )
        except FeatureNotFoundError:
            raise ValueError(
                "No feature versions recorded yet. "
                "Run 'metaxy graph push' first to record the feature graph snapshot."
            )
    else:
        print(f"From: snapshot {from_snapshot_version}...")

    # Step 2: Determine to_graph and to_snapshot_version
    if to_snapshot_version is None:
        # Default mode: record current active graph and use its snapshot
        # This ensures the to_snapshot is available in the store for comparison
        snapshot_result = store.record_feature_graph_snapshot()
        to_snapshot_version = snapshot_result.snapshot_version
        was_already_recorded = snapshot_result.already_recorded
        to_graph = FeatureGraph.get_active()
        if was_already_recorded:
            print(
                f"To: current active graph (snapshot {to_snapshot_version}... already recorded)"
            )
        else:
            print(
                f"To: current active graph (snapshot {to_snapshot_version}... recorded)"
            )

    else:
        # Historical mode: load from snapshot with force_reload
        # force_reload ensures we get current code from disk, not cached imports
        # Need to load the snapshot data from the store first
        from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

        feature_versions = store.read_metadata(FEATURE_VERSIONS_KEY, current_only=False)
        snapshot_data_df = nw.from_native(
            feature_versions.filter(
                nw.col("snapshot_version") == to_snapshot_version
            ).collect()
        )

        if snapshot_data_df.shape[0] == 0:
            raise ValueError(f"Snapshot {to_snapshot_version} not found in store")

        # Reconstruct snapshot data from the feature versions table
        snapshot_data = {}
        for row in snapshot_data_df.iter_rows(named=True):
            feature_key_str = row["feature_key"]
            snapshot_data[feature_key_str] = {
                "feature_spec": row.get(
                    "feature_spec", {}
                ),  # This would need the actual spec
                "feature_version": row["feature_version"],
                "feature_spec_version": row["feature_spec_version"],
                "feature_class_path": row.get("feature_class_path", ""),
            }

        to_graph = FeatureGraph.from_snapshot(
            snapshot_data, class_path_overrides=class_path_overrides, force_reload=True
        )
        print(f"To: snapshot {to_snapshot_version}...")

    # Step 3: Detect changes by comparing snapshot_versions directly
    # We don't reconstruct from_graph - just compare snapshot_versions from the store
    # This avoids issues with stale cached imports when files have changed
    assert from_snapshot_version is not None, "from_snapshot_version must be set by now"
    assert to_snapshot_version is not None, "to_snapshot_version must be set by now"

    # Use GraphDiffer to detect changes
    differ = GraphDiffer()

    # Load snapshot data using GraphDiffer
    try:
        from_snapshot_data = differ.load_snapshot_data(store, from_snapshot_version)
    except ValueError:
        # Snapshot not found - nothing to migrate from
        print("No from_snapshot found in store.")
        return None

    # Build snapshot data for to_snapshot
    to_snapshot_data = to_graph.to_snapshot()

    # Compute GraphDiff using GraphDiffer
    graph_diff = differ.diff(
        from_snapshot_data,
        to_snapshot_data,
        from_snapshot_version,
        to_snapshot_version,
    )

    # Check if there are any changes
    if not graph_diff.has_changes:
        print("No feature changes detected. All features up to date!")
        return None

    # Create operations for root changed features
    root_operations = []
    for node in graph_diff.changed_nodes:
        feature_key_str = node.feature_key.to_string()
        feature_key_str.replace("/", "_")

        root_operations.append(
            DataVersionReconciliation(
                type="metaxy.migrations.ops.DataVersionReconciliation"
            )
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
                filters=[nw.col("snapshot_version") == from_snapshot_version],
            )
            # Only collect head(1) to check existence
            from_metadata_sample = nw.from_native(from_metadata.head(1).collect())
            if from_metadata_sample.shape[0] == 0:
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
            f"Reconcile data_versions due to changes in: {affected_by[0]}"
        else:
            (f"Reconcile data_versions due to changes in: {', '.join(affected_by)}")

        # Create operation (feature versions derived from snapshots)
        # DataVersionReconciliation doesn't have id, feature_key, or reason params
        # It only has a type field since it applies to all affected features
        downstream_operations.append(
            DataVersionReconciliation(
                type="metaxy.migrations.ops.DataVersionReconciliation"
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
    from metaxy.metadata_store.system_tables import MIGRATION_EVENTS_KEY

    parent_migration_id = None
    try:
        existing_migrations = store.read_metadata(
            MIGRATION_EVENTS_KEY, current_only=False
        )
        # Get most recent migration by timestamp - only collect the top row
        latest = nw.from_native(
            existing_migrations.sort("timestamp", descending=True).head(1).collect()
        )
        if latest.shape[0] > 0:
            parent_migration_id = latest["migration_id"][0]
    except FeatureNotFoundError:
        # No migrations yet
        pass

    # Note: from_snapshot_version and to_snapshot_version were already resolved earlier

    # Create migration (serialize operations to dicts)
    len(root_operations)

    # DiffMigration expects 'ops' as list of dicts with 'type' field
    # Since all operations are DataVersionReconciliation, create a single operation dict
    ops = [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}]

    migration = DiffMigration(
        migration_id=migration_id,
        parent=parent_migration_id or "initial",
        from_snapshot_version=from_snapshot_version,
        to_snapshot_version=to_snapshot_version,
        created_at=timestamp,
        ops=ops,
    )

    return migration
