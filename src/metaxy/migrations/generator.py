"""Migration file generation."""

import os
from datetime import datetime
from typing import TYPE_CHECKING

from metaxy.metadata_store.exceptions import FeatureNotFoundError
from metaxy.migrations.detector import detect_feature_changes
from metaxy.migrations.models import Migration
from metaxy.migrations.ops import DataVersionReconciliation
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.feature import FeatureRegistry


def _is_upstream_of(
    upstream_key: FeatureKey, downstream_key: FeatureKey, registry: "FeatureRegistry"
) -> bool:
    """Check if upstream_key is in the dependency chain of downstream_key.

    Args:
        upstream_key: Potential upstream feature
        downstream_key: Feature to check dependencies for
        registry: Feature registry

    Returns:
        True if upstream_key is a direct or transitive dependency of downstream_key
    """
    plan = registry.get_feature_plan(downstream_key)

    if plan.deps is None:
        return False

    # Check direct dependencies
    for dep in plan.deps:
        if dep.key == upstream_key:
            return True

    # Check transitive dependencies (recursive)
    for dep in plan.deps:
        if _is_upstream_of(upstream_key, dep.key, registry):
            return True

    return False


def generate_migration(
    store: "MetadataStore",
    *,
    output_dir: str = "migrations",
) -> str | None:
    """Generate migration file from detected feature changes.

    Automatically detects all features that have changed (comparing code vs metadata)
    and generates explicit operations for ALL affected features (root + downstream).

    Each downstream feature gets its own explicit DataVersionReconciliation operation
    with from=current_version, to=current_version (just reconciling data_versions).

    Args:
        store: Metadata store to check
        output_dir: Directory to write migration file (default: "migrations")

    Returns:
        Path to generated migration file, or None if no changes detected

    Example:
        >>> migration_file = generate_migration(store)

        Detected 2 root feature changes:
          ✓ video_processing: abc12345 → def67890
          ✓ audio_processing: xyz11111 → xyz22222

        Generating explicit operations for 5 downstream features:
          ✓ feature_c (current: aaa111)
          ✓ feature_d (current: bbb222)
          ✓ feature_e (current: ccc333)

        Generated 7 total operations (2 root + 5 downstream)

        Generated: migrations/20250113_103000_update_video_processing_and_1_more.yaml
    """
    from metaxy.models.feature import FeatureRegistry

    registry = FeatureRegistry.get_active()

    # Detect root changes
    root_operations = detect_feature_changes(store)

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
        print(f"  ✓ {feature_key_str}: {op.from_} → {op.to}")

    # Discover downstream features that need reconciliation
    root_keys = [FeatureKey(op.feature_key) for op in root_operations]
    downstream_keys = registry.get_downstream_features(root_keys)

    # Create explicit operations for downstream features
    downstream_operations = []

    if downstream_keys:
        print(
            f"\nGenerating explicit operations for {len(downstream_keys)} downstream feature(s):"
        )

    for downstream_key in downstream_keys:
        feature_key_str = downstream_key.to_string()
        feature_cls = registry.features_by_key[downstream_key]

        # Query current feature_version from metadata
        try:
            current_metadata = store.read_metadata(
                feature_cls, current_only=True, allow_fallback=False
            )
            if len(current_metadata) > 0:
                current_version = current_metadata["feature_version"][0]
            else:
                # No current metadata, skip
                print(f"  ⊘ {feature_key_str} (not materialized yet, skipping)")
                continue
        except FeatureNotFoundError:
            # Feature not materialized yet
            print(f"  ⊘ {feature_key_str} (not materialized yet, skipping)")
            continue

        # Determine which root changes affect this downstream feature
        registry.get_feature_plan(downstream_key)
        affected_by = []

        for root_op in root_operations:
            root_key = FeatureKey(root_op.feature_key)
            # Check if this root is in the upstream dependency chain
            if _is_upstream_of(root_key, downstream_key, registry):
                affected_by.append(root_key.to_string())

        # Build informative reason
        if len(affected_by) == 1:
            reason = f"Reconcile data_versions due to changes in: {affected_by[0]}"
        else:
            reason = (
                f"Reconcile data_versions due to changes in: {', '.join(affected_by)}"
            )

        # Create operation with from=current, to=current (just reconciling data_versions)
        feature_key_slug = feature_key_str.replace("/", "_")
        op_id = f"reconcile_{feature_key_slug}_{current_version[:4]}"

        downstream_operations.append(
            DataVersionReconciliation(
                id=op_id,
                feature_key=list(downstream_key),
                from_=current_version,
                to=current_version,  # Same! Just reconciling data_versions
                reason=reason,
            )
        )

        print(f"  ✓ {feature_key_str} (current: {current_version})")

    # Combine all operations
    all_operations = root_operations + downstream_operations

    print(
        f"\nGenerated {len(all_operations)} total operations "
        f"({len(root_operations)} root + {len(downstream_operations)} downstream)"
    )

    # Build filename from root feature names
    root_count = len(root_operations)
    feature_names = "_".join(
        FeatureKey(op.feature_key).to_string().replace("/", "_")
        for op in root_operations[:2]
    )
    if root_count > 2:
        feature_names += f"_and_{root_count - 2}_more"

    filename = f"{output_dir}/{timestamp_str}_update_{feature_names}.yaml"

    # Find the latest migration to set as parent
    from metaxy.migrations.executor import MIGRATIONS_KEY
    from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

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

    # Get current snapshot_id (latest recorded)
    try:
        feature_versions = store.read_metadata(FEATURE_VERSIONS_KEY, current_only=False)
        if len(feature_versions) > 0:
            # Get most recent snapshot
            latest_snapshot = feature_versions.sort("recorded_at", descending=True).head(1)
            current_snapshot_id = latest_snapshot["snapshot_id"][0]
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

    # Create migration (serialize operations to dicts)
    migration = Migration(
        version=1,
        id=migration_id,
        parent_migration_id=parent_migration_id,
        snapshot_id=current_snapshot_id,
        description=f"Auto-generated migration for {root_count} changed feature(s) + {len(downstream_operations)} downstream",
        created_at=timestamp,
        operations=[op.model_dump(by_alias=True) for op in all_operations],
    )

    # Write YAML file
    os.makedirs(output_dir, exist_ok=True)
    migration.to_yaml(filename)

    # Print next steps
    print(f"\nGenerated: {filename}")
    print("\nNEXT STEPS:")
    print("1. Review the migration file")
    print("2. Edit the 'reason' fields for root changes")
    print("3. Run 'metaxy migrations apply --dry-run' to preview")
    print("4. Run 'metaxy migrations apply' to execute")
    print("5. Commit the migration file to git")

    return filename
