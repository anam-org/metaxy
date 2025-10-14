"""Feature change detection for automatic migration generation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY
from metaxy.metadata_store.exceptions import FeatureNotFoundError
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore


@dataclass
class FeatureChange:
    """Detected change in a feature version."""

    feature_key: FeatureKey
    from_version: str  # Old feature_version from metadata
    to_version: str  # New feature_version from code
    change_type: str  # "code_version", "dependencies", "containers", "unknown"


def detect_feature_changes(store: "MetadataStore") -> list[FeatureChange]:
    """Auto-detect all features that need migration.

    Compares current feature definitions (in code registry) with feature_versions
    in the metadata store. Automatically selects the latest (most recent) old
    version for each changed feature.

    Args:
        store: Metadata store to check

    Returns:
        List of detected changes, empty if all features are up to date

    Example:
        >>> changes = detect_feature_changes(store)
        >>> for change in changes:
        ...     print(f"{change.feature_key.to_string()}: {change.from_version} → {change.to_version}")
        video_processing: abc12345 → def67890
        audio_processing: xyz11111 → xyz22222
    """
    import polars as pl

    registry = store.registry
    changes = []

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

        # Feature changed! Use latest version as migration source
        changes.append(
            FeatureChange(
                feature_key=feature_key,
                from_version=latest_version,
                to_version=current_version,
                change_type="unknown",  # Could enhance to detect specific changes
            )
        )

    return changes
