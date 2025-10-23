"""Feature change detection for automatic migration generation."""

from typing import TYPE_CHECKING

from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY
from metaxy.metadata_store.exceptions import FeatureNotFoundError
from metaxy.migrations.ops import DataVersionReconciliation

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.feature import FeatureGraph


def detect_feature_changes(
    store: "MetadataStore",
    from_graph: "FeatureGraph",
    to_graph: "FeatureGraph",
) -> list[DataVersionReconciliation]:
    """Detect feature changes by comparing two feature graphs.

    Pure comparison function that compares feature versions between two graphs
    by querying their snapshot metadata from the store.
    
    Returns operations ONLY for features where feature_version changed but
    computation is unchanged (refactoring, dependency improvements, schema changes).
    
    For actual computation changes (new models, different algorithms), users must
    re-run their pipeline - migrations cannot recreate lost computation.

    Args:
        store: Metadata store containing snapshot metadata
        from_graph: Source feature graph (old state)
        to_graph: Target feature graph (new state)

    Returns:
        List of DataVersionReconciliation operations for changed features

    Example:
        >>> # Compare latest snapshot in store vs current code
        >>> from_graph = load_latest_snapshot(store)
        >>> to_graph = FeatureGraph.get_active()
        >>> operations = detect_feature_changes(store, from_graph, to_graph)
        >>> for op in operations:
        ...     print(f"Changed: {op.feature_key} - {op.reason}")
    """
    import polars as pl

    from_snapshot_id = from_graph.snapshot_id
    to_snapshot_id = to_graph.snapshot_id

    operations = []

    # Query feature versions for both snapshots
    try:
        from_versions = store.read_metadata(
            FEATURE_VERSIONS_KEY,
            current_only=False,
            allow_fallback=False,
            filters=pl.col("snapshot_id") == from_snapshot_id,
        )
    except FeatureNotFoundError:
        # No from_snapshot - nothing to migrate from
        return []

    try:
        to_versions = store.read_metadata(
            FEATURE_VERSIONS_KEY,
            current_only=False,
            allow_fallback=False,
            filters=pl.col("snapshot_id") == to_snapshot_id,
        )
    except FeatureNotFoundError:
        # No to_snapshot - nothing to migrate to
        return []

    # Build lookup dictionaries: feature_key -> feature_version
    from_versions_dict = {
        row["feature_key"]: row["feature_version"]
        for row in from_versions.iter_rows(named=True)
    }
    to_versions_dict = {
        row["feature_key"]: row["feature_version"]
        for row in to_versions.iter_rows(named=True)
    }

    # Check each feature in to_graph (target state)
    for feature_key, feature_cls in to_graph.features_by_key.items():
        feature_key_str = feature_key.to_string()

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
        feature_key_slug = feature_key_str.replace("/", "_")
        op_id = f"reconcile_{feature_key_slug}"

        operations.append(
            DataVersionReconciliation(
                id=op_id,
                feature_key=list(feature_key),
                reason="TODO: Describe what changed and why",
            )
        )

    return operations
