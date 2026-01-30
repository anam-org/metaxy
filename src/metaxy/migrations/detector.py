"""Feature change detection for automatic migration generation."""

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from metaxy._hashing import ensure_hash_compatibility, get_hash_truncation_length
from metaxy.graph.diff.differ import GraphDiffer
from metaxy.migrations.models import DiffMigration, FullGraphMigration
from metaxy.models.feature import FeatureGraph

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore


def detect_diff_migration(
    store: "MetadataStore",
    project: str | None = None,
    from_snapshot_version: str | None = None,
    ops: list[dict[str, Any]] | None = None,
    migrations_dir: Path | None = None,
    name: str | None = None,
    command: str | None = None,
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
        command: CLI command that generated this migration (written as YAML comment)

    Returns:
        DiffMigration if changes detected and written, None otherwise

    Example:
        <!-- skip next -->
        ```py
        # Compare latest snapshot in store vs current graph
        with store:
            migration = detect_diff_migration(store, project="my_project")
            if migration:
                print(f"Migration written to {migration.yaml_path}")
        ```

        <!-- skip next -->
        ```py
        # Use custom operation
        migration = detect_diff_migration(store, project="my_project", ops=[{"type": "myproject.ops.CustomOp"}])
        ```

        <!-- skip next -->
        ```py
        # Use custom name
        migration = detect_diff_migration(store, project="my_project", name="example_migration")
        ```
    """
    differ = GraphDiffer()

    # Get from_snapshot_version (use latest if not specified)
    if from_snapshot_version is None:
        from metaxy.metadata_store.system.storage import SystemTableStorage

        with store:
            storage = SystemTableStorage(store)
            snapshots = storage.read_graph_snapshots(project=project)
        if snapshots.height == 0:
            # No snapshots in store for this project - nothing to migrate from
            return None
        from_snapshot_version = snapshots["metaxy_snapshot_version"][0]

    # At this point, from_snapshot_version is guaranteed to be a str
    assert from_snapshot_version is not None  # Type narrowing for type checker

    # Get to_snapshot_version from current active graph
    active_graph = FeatureGraph.get_active()
    if len(active_graph.feature_definitions_by_key) == 0:
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
        # Write command as a comment header if provided
        if command:
            f.write(f"# Generated by: {command}\n")
        yaml.safe_dump(migration_yaml, f, sort_keys=False, default_flow_style=False)

    return migration


def generate_full_graph_migration(
    store: "MetadataStore",
    project: str | None = None,
    ops: list[dict[str, Any]] | None = None,
    migrations_dir: Path | None = None,
    name: str | None = None,
    command: str | None = None,
) -> "FullGraphMigration":
    """Generate a FullGraphMigration that includes all features in the current graph.

    Creates a migration YAML file with all feature keys specified in each operation's
    'features' list.

    Args:
        store: Metadata store (used to push snapshot)
        project: Project name
        ops: List of operation dicts with "type" field
        migrations_dir: Directory to write migration YAML (defaults to .metaxy/migrations/)
        name: Migration name (creates {timestamp}_{name} ID and filename)
        command: CLI command that generated this migration (written as YAML comment)

    Returns:
        FullGraphMigration with all features

    Raises:
        ValueError: If no features in active graph or ops not provided
    """
    from metaxy.metadata_store.system.storage import SystemTableStorage

    # Get active graph
    active_graph = FeatureGraph.get_active()
    if len(active_graph.feature_definitions_by_key) == 0:
        raise ValueError("No features in active graph")

    # Get all feature keys in topological order
    all_feature_keys = active_graph.topological_sort_features(list(active_graph.feature_definitions_by_key.keys()))
    feature_key_strings = [key.to_string() for key in all_feature_keys]

    # ops is required
    if ops is None or len(ops) == 0:
        raise ValueError(
            "ops parameter is required - must explicitly specify migration operations. "
            "Example: ops=[{'type': 'myproject.ops.CustomBackfill'}]"
        )

    # Add features to each operation
    ops_with_features = []
    for op in ops:
        op_copy = dict(op)
        op_copy["features"] = feature_key_strings
        ops_with_features.append(op_copy)

    # Push snapshot to get the current snapshot version
    # Resolve project - use provided or infer from first feature
    if project is None:
        project = next(iter(active_graph.feature_definitions_by_key.values())).project

    with store:
        storage = SystemTableStorage(store)
        snapshot_result = storage.push_graph_snapshot(project=project)
        snapshot_version = snapshot_result.snapshot_version

    # Generate migration ID (timestamp first for sorting)
    timestamp = datetime.now(timezone.utc)
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    if name is not None:
        migration_id = f"{timestamp_str}_{name}"
    else:
        migration_id = f"{timestamp_str}"

    # Default migrations directory
    if migrations_dir is None:
        migrations_dir = Path(".metaxy/migrations")

    migrations_dir.mkdir(parents=True, exist_ok=True)

    # Find parent migration (latest migration in chain)
    from metaxy.migrations.loader import find_latest_migration

    parent = find_latest_migration(migrations_dir)
    if parent is None:
        parent = "initial"

    # Create FullGraphMigration
    migration = FullGraphMigration(
        migration_id=migration_id,
        created_at=timestamp,
        parent=parent,
        snapshot_version=snapshot_version,
        ops=ops_with_features,
    )

    # Write migration YAML file
    import yaml

    yaml_path = migrations_dir / f"{migration_id}.yaml"
    migration_yaml = {
        "migration_type": "metaxy.migrations.models.FullGraphMigration",
        "id": migration.migration_id,
        "created_at": migration.created_at.isoformat(),
        "parent": migration.parent,
        "snapshot_version": migration.snapshot_version,
        "ops": migration.ops,
    }

    with open(yaml_path, "w") as f:
        # Write command as a comment header if provided
        if command:
            f.write(f"# Generated by: {command}\n")
        yaml.safe_dump(migration_yaml, f, sort_keys=False, default_flow_style=False)

    return migration
