"""Feature change detection for automatic migration generation."""

from typing import TYPE_CHECKING

import narwhals as nw

from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY
from metaxy.metadata_store.exceptions import FeatureNotFoundError
from metaxy.migrations.ops import DataVersionReconciliation
from metaxy.models.types import FEATURE_KEY_SEPARATOR

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore


def detect_feature_changes(
    store: "MetadataStore",
    from_snapshot_version: str,
    to_snapshot_version: str,
) -> list[DataVersionReconciliation]:
    """Detect feature changes by comparing snapshot_versions directly.

    Pure comparison function that compares feature versions between two snapshots
    by querying their snapshot metadata from the store.

    This does NOT reconstruct graphs, avoiding issues with:
    - Importing stale cached code when files have changed
    - Need for custom methods during detection (only needed for execution)

    For actual computation changes (new models, different algorithms), users must
    re-run their pipeline - migrations cannot recreate lost computation.

    Args:
        store: Metadata store containing snapshot metadata
        from_snapshot_version: Source snapshot version (old state)
        to_snapshot_version: Target snapshot version (new state)

    Returns:
        List of DataVersionReconciliation operations for changed features

    Example:
        >>> # Compare latest snapshot in store vs current snapshot
        >>> operations = detect_feature_changes(store, old_snapshot_version, new_snapshot_version)
        >>> for op in operations:
        ...     print(f"Changed: {op.feature_key} - {op.reason}")
    """
    operations = []

    # Query feature versions for both snapshots
    try:
        from_versions = store.read_metadata(
            FEATURE_VERSIONS_KEY,
            current_only=False,
            allow_fallback=False,
            filters=[nw.col("snapshot_version") == from_snapshot_version],
        )
    except FeatureNotFoundError:
        # No from_snapshot - nothing to migrate from
        return []

    try:
        to_versions = store.read_metadata(
            FEATURE_VERSIONS_KEY,
            current_only=False,
            allow_fallback=False,
            filters=[nw.col("snapshot_version") == to_snapshot_version],
        )
    except FeatureNotFoundError:
        # No to_snapshot - nothing to migrate to
        return []

    # Collect to iterate (we need to build dictionaries)
    from_versions_eager = nw.from_native(from_versions.collect())
    to_versions_eager = nw.from_native(to_versions.collect())

    # Build lookup dictionaries: feature_key -> feature_version
    from_versions_dict = {
        row["feature_key"]: row["feature_version"]
        for row in from_versions_eager.iter_rows(named=True)
    }
    to_versions_dict = {
        row["feature_key"]: row["feature_version"]
        for row in to_versions_eager.iter_rows(named=True)
    }

    # Check each feature in to_versions (target state)
    for feature_key_str in to_versions_dict.keys():
        # Get versions from both snapshots
        from_version = from_versions_dict.get(feature_key_str)
        to_version = to_versions_dict.get(feature_key_str)

        if from_version is None:
            # Feature doesn't exist in from_snapshot - it's new, no migration needed
            continue

        if to_version is None:
            # Feature exists in from_snapshot but not in to_snapshot - it was removed
            # No migration needed (we don't migrate deleted features)
            continue

        if from_version == to_version:
            # Versions match - no migration needed
            continue

        # Feature changed! Create operation with auto-generated ID
        # Parse feature_key from string (e.g., "video__files" -> ["video", "files"])
        from metaxy.models.types import FeatureKey

        feature_key = FeatureKey(feature_key_str.split(FEATURE_KEY_SEPARATOR))
        op_id = f"reconcile_{feature_key_str}"

        operations.append(
            DataVersionReconciliation(
                id=op_id,
                feature_key=list(feature_key),
                reason="TODO: Describe what changed and why",
            )
        )

    return operations
