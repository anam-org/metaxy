"""Feature change detection for automatic migration generation."""

from typing import TYPE_CHECKING

from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY
from metaxy.metadata_store.exceptions import FeatureNotFoundError
from metaxy.migrations.ops import DataVersionReconciliation

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore


def detect_feature_changes(
    store: "MetadataStore",
) -> list[DataVersionReconciliation]:
    """Auto-detect all features that need migration.

    Compares current feature definitions (in code registry) with feature_versions
    in the metadata store. Automatically selects the latest (most recent) old
    version for each changed feature.

    Returns DataVersionReconciliation operations ready to use in migrations.

    Args:
        store: Metadata store to check

    Returns:
        List of DataVersionReconciliation operations, empty if all features are up to date

    Example:
        >>> operations = detect_feature_changes(store)
        >>> for op in operations:
        ...     print(f"{op.id}: {FeatureKey(op.feature_key).to_string()} {op.from_} → {op.to}")
        reconcile_video_processing_abc1_to_def6: video_processing abc12345 → def67890
        reconcile_audio_processing_xyz1_to_xyz2: audio_processing xyz11111 → xyz22222
    """
    import polars as pl

    from metaxy.models.feature import FeatureRegistry

    operations = []

    # Get the active registry
    registry = FeatureRegistry.get_active()

    # Query ALL feature version history once (efficient)
    try:
        all_version_history = store.read_metadata(
            FEATURE_VERSIONS_KEY,
            current_only=False,
        ).sort("recorded_at", descending=True)
    except FeatureNotFoundError:
        # No feature versions recorded yet - no migrations needed
        return []

    # Build lookup: feature_key -> latest_version
    # Group by feature_key and take first (most recent due to sort)
    latest_versions = all_version_history.group_by(
        "feature_key", maintain_order=True
    ).agg(pl.col("feature_version").first().alias("latest_version"))

    # Create dict for fast lookup
    latest_by_feature = {
        row["feature_key"]: row["latest_version"]
        for row in latest_versions.iter_rows(named=True)
    }

    # Check each feature in registry
    for feature_key, feature_cls in registry.features_by_key.items():
        feature_key_str = feature_key.to_string()

        # Get current version from code
        current_version = feature_cls.feature_version()  # type: ignore[attr-defined]

        # Look up latest version from metadata (fast dict lookup)
        latest_version = latest_by_feature.get(feature_key_str)

        if latest_version is None:
            # Feature never materialized - no migration needed
            continue

        if current_version == latest_version:
            # Current version matches latest - no migration needed
            continue

        # Feature changed! Create operation with auto-generated ID
        feature_key_slug = feature_key_str.replace("/", "_")
        op_id = f"reconcile_{feature_key_slug}_{latest_version[:4]}_to_{current_version[:4]}"

        operations.append(
            DataVersionReconciliation(
                id=op_id,
                feature_key=list(feature_key),
                from_=latest_version,
                to=current_version,
                reason="TODO: Describe what changed and why",
            )
        )

    return operations
