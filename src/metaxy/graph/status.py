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


class SamplePreviews(BaseModel):
    """Preview data for added and changed samples.

    This model separates data collection from rendering, making it
    reusable for different display contexts (CLI, graph viz, etc.).
    """

    model_config = {"arbitrary_types_allowed": True}

    added: pl.DataFrame | None = Field(
        default=None, description="Preview of added samples"
    )
    changed: pl.DataFrame | None = Field(
        default=None, description="Preview of changed samples"
    )


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

    def collect_sample_previews(
        self,
        id_columns: Sequence[str] | None = None,
        limit: int = 5,
    ) -> SamplePreviews:
        """Collect preview data for added and changed samples.

        This separates data collection from formatting for reusability.

        Args:
            id_columns: Columns to include in previews (defaults to ["sample_uid"])
            limit: Maximum number of samples to preview per category

        Returns:
            SamplePreviews with DataFrames for added/changed samples
        """
        if self.lazy_increment is None:
            return SamplePreviews()

        display_columns = list(id_columns) if id_columns else ["sample_uid"]

        added = None
        if self.added_count > 0:
            added = (
                self.lazy_increment.added.select(display_columns)
                .head(limit)
                .collect()
                .to_polars()
            )

        changed = None
        if self.changed_count > 0:
            changed = (
                self.lazy_increment.changed.select(display_columns)
                .head(limit)
                .collect()
                .to_polars()
            )

        return SamplePreviews(added=added, changed=changed)

    def format_sample_previews(self, previews: SamplePreviews) -> list[str]:
        """Format pre-collected sample previews for display.

        This method only handles formatting. Use collect_sample_previews()
        to get the preview data first.

        Args:
            previews: Pre-collected preview data from collect_sample_previews()

        Returns:
            List of formatted preview lines (empty if no data in previews)

        Example:
            ```python
            # Collect once, format multiple times or reuse for viz
            previews = status.collect_sample_previews(id_columns=["sample_uid"])

            # Format for CLI display
            cli_lines = status.format_sample_previews(previews)
            for line in cli_lines:
                console.print(line)

            # Reuse for graph visualization
            render_graph_node(node_id, previews)
            ```
        """
        lines: list[str] = []

        if previews.added is not None and previews.added.height > 0:
            preview_lines = [
                ", ".join(f"{col}={row[col]}" for col in previews.added.columns)
                for row in previews.added.to_dicts()
            ]
            lines.append("    Added samples: " + "; ".join(preview_lines))

        if previews.changed is not None and previews.changed.height > 0:
            preview_lines = [
                ", ".join(f"{col}={row[col]}" for col in previews.changed.columns)
                for row in previews.changed.to_dicts()
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
        use_fallback: Whether to read from fallback stores when checking status.
            Defaults to False to only check the primary store.
            Set to True to include fallback stores in status checks.

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
    try:
        lazy_increment = metadata_store.resolve_update(
            feature_cls,
            lazy=True,
            allow_fallback=use_fallback,
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
            columns=list(id_columns_seq) if id_columns_seq is not None else None,
            allow_fallback=use_fallback,
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
