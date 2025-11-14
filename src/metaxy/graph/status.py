"""Feature metadata status inspection utilities.

This module provides reusable SDK functions for inspecting feature metadata status,
useful for both CLI commands and programmatic usage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import narwhals as nw
import polars as pl

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.feature import BaseFeature


def count_lazy_rows(lazy_frame: nw.LazyFrame[Any]) -> int:
    """Return row count for a Narwhals LazyFrame.

    Args:
        lazy_frame: The LazyFrame to count rows from

    Returns:
        Number of rows in the LazyFrame
    """
    return lazy_frame.select(nw.len()).collect().to_polars()["len"].item()


def preview_samples(
    lazy_frame: nw.LazyFrame[Any],
    id_columns: Sequence[str] | None = None,
    limit: int = 5,
) -> pl.DataFrame:
    """Return preview of samples as a Polars DataFrame.

    Args:
        lazy_frame: The LazyFrame containing samples
        id_columns: Columns to include in the preview (defaults to ["sample_uid"])
        limit: Maximum number of rows to include

    Returns:
        Polars DataFrame with the requested columns and row limit
    """
    # Determine columns to display
    headers = list(id_columns or ["sample_uid"])
    available_headers = [col for col in headers if col in lazy_frame.columns]

    if not available_headers:
        # Fallback to all columns if specified headers don't exist
        available_headers = lazy_frame.columns

    if not available_headers:
        return pl.DataFrame()

    # Only collect limited rows
    return lazy_frame.select(available_headers).head(limit).collect().to_polars()


def get_feature_metadata_status(
    feature: type[BaseFeature],
    metadata_store: MetadataStore,
) -> dict[str, Any]:
    """Get metadata status for a single feature.

    Args:
        feature: The feature class to check
        metadata_store: The metadata store to query

    Returns:
        Dictionary with status information:
        - 'feature_key': The feature key
        - 'target_version': The feature version from code
        - 'metadata_exists': Whether metadata exists in the store
        - 'row_count': Number of metadata rows (0 if none exist)
        - 'added_count': Number of samples that would be added
        - 'changed_count': Number of samples that would be changed
        - 'needs_update': Boolean indicating if updates are needed
        - 'lazy_increment': The LazyIncrement object (None if no metadata exists)
    """
    from metaxy.metadata_store.exceptions import FeatureNotFoundError

    feature_key = feature.spec().key  # type: ignore[attr-defined]
    target_version = feature.feature_version()

    # Try to get the increment
    try:
        lazy_increment = metadata_store.resolve_update(feature, lazy=True)
    except FeatureNotFoundError:
        # No metadata exists at all
        return {
            "feature_key": feature_key,
            "target_version": target_version,
            "metadata_exists": False,
            "row_count": 0,
            "added_count": 0,
            "changed_count": 0,
            "needs_update": True,
            "lazy_increment": None,
        }

    # Count changes
    added_count = count_lazy_rows(lazy_increment.added)
    changed_count = count_lazy_rows(lazy_increment.changed)

    # Get row count for this feature version
    id_columns = feature.spec().id_columns  # type: ignore[attr-defined]
    id_columns_seq = tuple(id_columns) if id_columns is not None else None

    try:
        metadata_lazy = metadata_store.read_metadata(
            feature_key,
            feature_version=target_version,
            current_only=False,
            columns=list(id_columns_seq) if id_columns_seq is not None else None,
        )
        row_count = count_lazy_rows(metadata_lazy)
    except FeatureNotFoundError:
        row_count = 0

    return {
        "feature_key": feature_key,
        "target_version": target_version,
        "metadata_exists": True,
        "row_count": row_count,
        "added_count": added_count,
        "changed_count": changed_count,
        "needs_update": added_count > 0 or changed_count > 0,
        "lazy_increment": lazy_increment,
    }
