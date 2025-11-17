"""Feature change detection for automatic migration generation."""

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from metaxy.graph.diff.differ import GraphDiffer
from metaxy.migrations.models import DiffMigration
from metaxy.models.feature import FeatureGraph
from metaxy.utils.hashing import ensure_hash_compatibility, get_hash_truncation_length

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore


def detect_diff_migration(
    store: "MetadataStore",
    project: str | None = None,
    from_snapshot_version: str | None = None,
    ops: list[dict[str, Any]] | None = None,
    migrations_dir: Path | None = None,
    name: str | None = None,
) -> "DiffMigration | None":
    """Detect migration needed between snapshots and write YAML file.

    Compares the latest snapshot in the store (or specified from_snapshot_version)
    with the current active graph to detect changes and generate a migration YAML file.

    Args:
        store: Metadata store containing snapshot metadata
        project: Project name for filtering snapshots
        from_snapshot_version: Source snapshot version (defaults to latest in store for project)
        ops: List of operation dicts with "type" field (defaults to [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}])
        migrations_dir: Directory to write migration YAML (defaults to .metaxy/migrations/)
        name: Migration name (creates {timestamp}_{name} ID and filename)

    Returns:
        DiffMigration if changes detected and written, None otherwise

    Example:
        ```py
        # Compare latest snapshot in store vs current graph
        with store:
            migration = detect_diff_migration(store, project="my_project")
            if migration:
            print(f"Migration written to {migration.yaml_path}")

        ```py
        # Use custom operation
        migration = detect_diff_migration(store, project="my_project", ops=[{"type": "myproject.ops.CustomOp"}])
        ```

        ```py
        # Use custom name
        migration = detect_diff_migration(store, project="my_project", name="example_migration")
        ```
    """
    differ = GraphDiffer()

    # Get from_snapshot_version (use latest if not specified)
    if from_snapshot_version is None:
        snapshots = store.read_graph_snapshots(project=project)
        if snapshots.height == 0:
            # No snapshots in store for this project - nothing to migrate from
            return None
        from_snapshot_version = snapshots["metaxy_snapshot_version"][0]

    # At this point, from_snapshot_version is guaranteed to be a str
    assert from_snapshot_version is not None  # Type narrowing for type checker

    # Get to_snapshot_version from current active graph
    active_graph = FeatureGraph.get_active()
    if len(active_graph.features_by_key) == 0:
        # No features in active graph - nothing to migrate to
        return None

    to_snapshot_version = active_graph.snapshot_version

    # Check hash truncation compatibility
    # If truncation is in use, the snapshot versions should be compatible
    # (either exactly equal or one is a truncated version of the other)
    truncation_length = get_hash_truncation_length()
    if truncation_length is not None:
        # When using truncation, we need to check compatibility rather than exact equality
        if ensure_hash_compatibility(from_snapshot_version, to_snapshot_version):
            # Hashes are compatible (same or truncated versions) - no changes
            return None
    else:
        # No truncation - use exact comparison
        if from_snapshot_version == to_snapshot_version:
            return None

    # Load snapshot data using GraphDiffer
    try:
        from_snapshot_data = differ.load_snapshot_data(store, from_snapshot_version)
    except ValueError:
        # Snapshot not found - nothing to migrate from
        return None

    # Build snapshot data for to_snapshot (current graph)
    to_snapshot_data = active_graph.to_snapshot()

    # Compute GraphDiff using GraphDiffer
    graph_diff = differ.diff(
        from_snapshot_data,
        to_snapshot_data,
        from_snapshot_version,
        to_snapshot_version,
    )

    # Check if there are any changes
    if not graph_diff.has_changes:
        return None

    # Generate migration ID (timestamp first for sorting)
    timestamp = datetime.now(timezone.utc)
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    if name is not None:
        migration_id = f"{timestamp_str}_{name}"
    else:
        migration_id = f"{timestamp_str}"

    # ops is required - caller must specify
    if ops is None:
        raise ValueError(
            "ops parameter is required - must explicitly specify migration operations. "
            "Example: ops=[{'type': 'metaxy.migrations.ops.DataVersionReconciliation'}]"
        )

    # Default migrations directory
    if migrations_dir is None:
        migrations_dir = Path(".metaxy/migrations")

    migrations_dir.mkdir(parents=True, exist_ok=True)

    # Find parent migration (latest migration in chain)
    from metaxy.migrations.loader import find_latest_migration

    parent = find_latest_migration(migrations_dir)
    if parent is None:
        parent = "initial"

    # Create minimal DiffMigration - affected_features and description are computed on-demand
    migration = DiffMigration(
        migration_id=migration_id,
        created_at=timestamp,
        parent=parent,
        from_snapshot_version=from_snapshot_version,
        to_snapshot_version=to_snapshot_version,
        ops=ops,
    )

    # Write migration YAML file
    import yaml

    yaml_path = migrations_dir / f"{migration_id}.yaml"
    migration_yaml = {
        "migration_type": "metaxy.migrations.models.DiffMigration",
        "id": migration.migration_id,
        "created_at": migration.created_at.isoformat(),
        "parent": migration.parent,
        "from_snapshot_version": migration.from_snapshot_version,
        "to_snapshot_version": migration.to_snapshot_version,
        "ops": migration.ops,
    }

    with open(yaml_path, "w") as f:
        yaml.safe_dump(migration_yaml, f, sort_keys=False, default_flow_style=False)

    return migration
