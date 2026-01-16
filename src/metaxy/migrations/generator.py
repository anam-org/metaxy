"""Migration generation."""

from datetime import datetime
from typing import TYPE_CHECKING

import narwhals as nw

from metaxy.graph.diff.differ import GraphDiffer
from metaxy.metadata_store.exceptions import FeatureNotFoundError
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.migrations.models import DiffMigration
from metaxy.migrations.ops import DataVersionReconciliation
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.feature import FeatureGraph


def _is_upstream_of(
    upstream_key: FeatureKey, downstream_key: FeatureKey, graph: "FeatureGraph"
) -> bool:
    """Check if upstream_key is in the dependency chain of downstream_key."""
    plan = graph.get_feature_plan(downstream_key)
    if plan.deps is None:
        return False

    for dep in plan.deps:
        if dep.key == upstream_key:
            return True
        if _is_upstream_of(upstream_key, dep.key, graph):
            return True
    return False


def _get_from_snapshot_version(store: "MetadataStore") -> str:
    """Get the from_snapshot_version from the store's latest snapshot."""
    from metaxy.metadata_store.system.keys import FEATURE_VERSIONS_KEY

    try:
        feature_versions = store.read_metadata(FEATURE_VERSIONS_KEY, current_only=False)
        latest_snapshot = nw.from_native(
            feature_versions.sort("recorded_at", descending=True).head(1).collect()
        )
        if latest_snapshot.shape[0] > 0:
            version = latest_snapshot["metaxy_snapshot_version"][0]
            print(f"From: latest snapshot {version}...")
            return version
        raise ValueError(
            "No feature graph snapshot found in metadata store. "
            "Run 'metaxy graph push' first to record feature versions before generating migrations."
        )
    except FeatureNotFoundError:
        raise ValueError(
            "No feature versions recorded yet. "
            "Run 'metaxy graph push' first to record the feature graph snapshot."
        )


def _get_to_graph_and_version(
    store: "MetadataStore",
    to_snapshot_version: str | None,
    class_path_overrides: dict[str, str] | None,
) -> tuple["FeatureGraph", str]:
    """Get the to_graph and to_snapshot_version."""
    from metaxy.models.feature import FeatureGraph

    if to_snapshot_version is None:
        snapshot_result = SystemTableStorage(store).push_graph_snapshot()
        to_snapshot_version = snapshot_result.snapshot_version
        was_already_pushed = snapshot_result.already_pushed
        to_graph = FeatureGraph.get_active()
        msg = "already pushed" if was_already_pushed else "pushed"
        print(f"To: current active graph (snapshot {to_snapshot_version}... {msg})")
    else:
        to_graph = SystemTableStorage(store).load_graph_from_snapshot(
            snapshot_version=to_snapshot_version,
            class_path_overrides=class_path_overrides,
            force_reload=True,
        )
        print(f"To: snapshot {to_snapshot_version}...")

    return to_graph, to_snapshot_version


def _create_downstream_operations(
    store: "MetadataStore",
    to_graph: "FeatureGraph",
    downstream_keys: list[FeatureKey],
    root_keys: list[FeatureKey],
    from_snapshot_version: str,
) -> list[DataVersionReconciliation]:
    """Create operations for downstream features."""
    downstream_operations = []

    if downstream_keys:
        print(
            f"\nGenerating explicit operations for {len(downstream_keys)} downstream feature(s):"
        )

    for downstream_key in downstream_keys:
        feature_key_str = downstream_key.to_string()
        feature_cls = to_graph.features_by_key[downstream_key]

        try:
            from_metadata = store.read_metadata(
                feature_cls,
                current_only=False,
                allow_fallback=False,
                filters=[nw.col("metaxy_snapshot_version") == from_snapshot_version],
            )
            from_metadata_sample = nw.from_native(from_metadata.head(1).collect())
            if from_metadata_sample.shape[0] == 0:
                print(f"  ⊘ {feature_key_str} (new feature, skipping)")
                continue
        except FeatureNotFoundError:
            print(f"  ⊘ {feature_key_str} (not materialized yet, skipping)")
            continue

        to_graph.get_feature_plan(downstream_key)
        for root_key in root_keys:
            if _is_upstream_of(root_key, downstream_key, to_graph):
                break

        downstream_operations.append(DataVersionReconciliation())
        print(f"  ✓ {feature_key_str}")

    return downstream_operations


def _find_parent_migration(store: "MetadataStore") -> str | None:
    """Find the latest migration to set as parent."""
    from metaxy.metadata_store.system import EVENTS_KEY

    try:
        existing_migrations = store.read_metadata(EVENTS_KEY, current_only=False)
        latest = nw.from_native(
            existing_migrations.sort("timestamp", descending=True).head(1).collect()
        )
        if latest.shape[0] > 0:
            return latest["migration_id"][0]
    except FeatureNotFoundError:
        pass
    return None


def generate_migration(
    store: "MetadataStore",
    *,
    project: str,
    from_snapshot_version: str | None = None,
    to_snapshot_version: str | None = None,
    class_path_overrides: dict[str, str] | None = None,
) -> DiffMigration | None:
    """Generate migration from detected feature changes or between snapshots.

    Two modes:
    1. Default mode (both snapshot_versions None): Compare latest snapshot vs current graph
    2. Historical mode (both provided): Compare two historical registries

    Args:
        store: Metadata store to check
        project: Project name for filtering snapshots
        from_snapshot_version: Optional snapshot version to compare from
        to_snapshot_version: Optional snapshot version to compare to
        class_path_overrides: Optional overrides for moved/renamed feature classes

    Returns:
        Migration object, or None if no changes detected
    """
    if from_snapshot_version is None:
        from_snapshot_version = _get_from_snapshot_version(store)
    else:
        print(f"From: snapshot {from_snapshot_version}...")

    to_graph, to_snapshot_version = _get_to_graph_and_version(
        store, to_snapshot_version, class_path_overrides
    )

    graph_diff = _compute_graph_diff(
        store, from_snapshot_version, to_snapshot_version, to_graph
    )
    if graph_diff is None or not graph_diff.has_changes:
        print("No feature changes detected. All features up to date!")
        return None

    root_keys = _get_root_feature_keys(graph_diff)
    if not root_keys:
        print("No feature changes detected. All features up to date!")
        return None

    _print_root_changes(root_keys)

    downstream_keys = to_graph.get_downstream_features(root_keys)
    downstream_operations = _create_downstream_operations(
        store, to_graph, downstream_keys, root_keys, from_snapshot_version
    )

    num_root = len(root_keys)
    num_downstream = len(downstream_operations)
    print(
        f"\nGenerated {num_root + num_downstream} total operations "
        f"({num_root} root + {num_downstream} downstream)"
    )

    return _build_migration(store, from_snapshot_version, to_snapshot_version)


def _compute_graph_diff(
    store: "MetadataStore",
    from_snapshot_version: str,
    to_snapshot_version: str,
    to_graph: "FeatureGraph",
):
    """Compute the graph diff between two snapshots."""
    differ = GraphDiffer()

    try:
        from_snapshot_data = differ.load_snapshot_data(store, from_snapshot_version)
    except ValueError:
        print("No from_snapshot found in store.")
        return None

    to_snapshot_data = to_graph.to_snapshot()
    return differ.diff(
        from_snapshot_data,
        to_snapshot_data,
        from_snapshot_version,
        to_snapshot_version,
    )


def _get_root_feature_keys(graph_diff) -> list[FeatureKey]:
    """Get feature keys for root changed features."""
    return [node.feature_key for node in graph_diff.changed_nodes]


def _print_root_changes(root_keys: list[FeatureKey]) -> None:
    """Print detected root changes."""
    print(f"\nDetected {len(root_keys)} root feature change(s):")
    for key in root_keys:
        print(f"  ✓ {key.to_string()}")


def _build_migration(
    store: "MetadataStore",
    from_snapshot_version: str,
    to_snapshot_version: str,
) -> DiffMigration:
    """Build the final migration object."""
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    migration_id = f"migration_{timestamp_str}"

    parent_migration_id = _find_parent_migration(store)
    ops = [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}]

    return DiffMigration(
        migration_id=migration_id,
        parent=parent_migration_id or "initial",
        from_snapshot_version=from_snapshot_version,
        to_snapshot_version=to_snapshot_version,
        created_at=timestamp,
        ops=ops,
    )
