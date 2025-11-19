"""Feature metadata status inspection utilities.

This module provides reusable SDK functions for inspecting feature metadata status,
useful for both CLI commands and programmatic usage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import narwhals as nw
import polars as pl
from pydantic import BaseModel, Field

from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.feature import BaseFeature
    from metaxy.versioning.types import LazyIncrement


class FeatureMetadataStatus(BaseModel):
    """Status information for feature metadata in a metadata store.

    This model encapsulates the current state of metadata for a feature,
    including whether it exists, needs updates, and sample counts.
    """

    feature_key: FeatureKey = Field(description="The feature key being inspected")
    target_version: str = Field(description="The feature version from code")
    metadata_exists: bool = Field(description="Whether metadata exists in the store")
    row_count: int = Field(description="Number of metadata rows (0 if none exist)")
    added_count: int = Field(description="Number of samples that would be added")
    changed_count: int = Field(description="Number of samples that would be changed")
    needs_update: bool = Field(description="Whether updates are needed")
    lazy_increment: LazyIncrement | None = Field(
        default=None,
        description="The LazyIncrement object (None if no metadata exists)",
        exclude=True,  # Exclude from serialization
    )

    def format_status_line(
        self,
        *,
        status_icon: str | None = None,
        status_text: str | None = None,
    ) -> str:
        """Format a status line for display.

        Args:
            status_icon: Optional custom icon (defaults to auto-detected icon)
            status_text: Optional custom text (defaults to auto-detected text)

        Returns:
            Formatted status line string
        """
        # Auto-detect status if not provided
        if status_icon is None or status_text is None:
            if not self.metadata_exists:
                status_icon = "[red]✗[/red]"
                status_text = "missing metadata"
            elif self.needs_update:
                status_icon = "[yellow]⚠[/yellow]"
                status_text = "needs update"
            else:
                status_icon = "[green]✓[/green]"
                status_text = "up-to-date"

        return (
            f"{status_icon} {self.feature_key.to_string()} "
            f"(rows: {self.row_count}, added: {self.added_count}, "
            f"changed: {self.changed_count}) — {status_text}"
        )

    def format_sample_previews(
        self,
        id_columns: Sequence[str] | None = None,
        limit: int = 5,
    ) -> list[str]:
        """Format sample previews for added and changed samples.

        Args:
            id_columns: Columns to include in previews (defaults to ["sample_uid"])
            limit: Maximum number of samples to preview per category

        Returns:
            List of formatted preview lines (empty if no lazy_increment)
        """
        if self.lazy_increment is None:
            return []

        lines: list[str] = []

        if self.added_count > 0:
            added_preview_df = preview_samples(
                self.lazy_increment.added,
                id_columns,
                limit,
            )
            if added_preview_df.height > 0:
                preview_lines = [
                    ", ".join(f"{col}={row[col]}" for col in added_preview_df.columns)
                    for row in added_preview_df.to_dicts()
                ]
                lines.append("    Added samples: " + "; ".join(preview_lines))

        if self.changed_count > 0:
            changed_preview_df = preview_samples(
                self.lazy_increment.changed,
                id_columns,
                limit,
            )
            if changed_preview_df.height > 0:
                preview_lines = [
                    ", ".join(f"{col}={row[col]}" for col in changed_preview_df.columns)
                    for row in changed_preview_df.to_dicts()
                ]
                lines.append("    Changed samples: " + "; ".join(preview_lines))

        return lines


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

    # Fallback to all columns if specified headers don't exist
    if not available_headers:
        available_headers = lazy_frame.columns

    # Only collect limited rows
    return lazy_frame.select(available_headers).head(limit).collect().to_polars()


def get_feature_metadata_status(
    feature_key: FeatureKey | type[BaseFeature],
    metadata_store: MetadataStore,
    *,
    use_fallback: bool = False,
) -> FeatureMetadataStatus:
    """Get metadata status for a single feature.

    Args:
        feature_key: The feature key or feature class to check
        metadata_store: The metadata store to query
        use_fallback: Whether to read from fallback stores (defaults to False)

    Returns:
        FeatureMetadataStatus with status information
    """
    from metaxy.metadata_store.exceptions import FeatureNotFoundError
    from metaxy.models.feature import BaseFeature, FeatureGraph

    # Handle both FeatureKey and feature class inputs
    if isinstance(feature_key, type) and issubclass(feature_key, BaseFeature):
        feature_cls = feature_key
        key = feature_cls.spec().key  # type: ignore[attr-defined]
    else:
        # feature_key is already a FeatureKey
        key = feature_key  # type: ignore[assignment]
        # Look up feature class from the active graph
        graph = FeatureGraph.get_active()
        if key not in graph.features_by_key:
            raise ValueError(f"Feature {key.to_string()} not found in active graph")
        feature_cls = graph.features_by_key[key]

    target_version = feature_cls.feature_version()

    # Try to get the increment
    # Note: use_fallback parameter is reserved for future use when fallback store
    # support is added to the metadata store methods
    try:
        lazy_increment = metadata_store.resolve_update(
            feature_cls,
            lazy=True,
        )
    except FeatureNotFoundError:
        # No metadata exists at all
        return FeatureMetadataStatus(
            feature_key=key,
            target_version=target_version,
            metadata_exists=False,
            row_count=0,
            added_count=0,
            changed_count=0,
            needs_update=True,
            lazy_increment=None,
        )

    # Count changes
    added_count = count_lazy_rows(lazy_increment.added)
    changed_count = count_lazy_rows(lazy_increment.changed)

    # Get row count for this feature version
    id_columns = feature_cls.spec().id_columns  # type: ignore[attr-defined]
    id_columns_seq = tuple(id_columns) if id_columns is not None else None

    try:
        metadata_lazy = metadata_store.read_metadata(
            key,
            feature_version=target_version,
            columns=list(id_columns_seq) if id_columns_seq is not None else None,
        )
        row_count = count_lazy_rows(metadata_lazy)
    except FeatureNotFoundError:
        row_count = 0

    return FeatureMetadataStatus(
        feature_key=key,
        target_version=target_version,
        metadata_exists=True,
        row_count=row_count,
        added_count=added_count,
        changed_count=changed_count,
        needs_update=added_count > 0 or changed_count > 0,
        lazy_increment=lazy_increment,
    )
