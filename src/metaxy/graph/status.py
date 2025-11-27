"""Feature metadata status inspection utilities.

This module provides reusable SDK functions for inspecting feature metadata status,
useful for both CLI commands and programmatic usage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import narwhals as nw
from pydantic import BaseModel, Field

from metaxy.models.types import (
    CoercibleToFeatureKey,
    FeatureKey,
    ValidatedFeatureKeyAdapter,
)

if TYPE_CHECKING:
    from metaxy import BaseFeature
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.versioning.types import LazyIncrement


class FullFeatureMetadataRepresentation(BaseModel):
    """Full JSON-safe representation of feature metadata status."""

    feature_key: str
    status: Literal["missing", "needs_update", "up_to_date", "root_feature"]
    needs_update: bool
    metadata_exists: bool
    rows: int
    added: int | None
    changed: int | None
    target_version: str
    is_root_feature: bool = False
    sample_details: list[str] | None = None


class FeatureMetadataStatus(BaseModel):
    """Status information for feature metadata in a metadata store.

    This model encapsulates the current state of metadata for a feature,
    including whether it exists, needs updates, and sample counts.

    This is a pure Pydantic model without arbitrary types. For working with
    LazyIncrement objects, use FeatureMetadataStatusWithIncrement.
    """

    feature_key: FeatureKey = Field(description="The feature key being inspected")
    target_version: str = Field(description="The feature version from code")
    metadata_exists: bool = Field(description="Whether metadata exists in the store")
    row_count: int = Field(description="Number of metadata rows (0 if none exist)")
    added_count: int = Field(description="Number of samples that would be added")
    changed_count: int = Field(description="Number of samples that would be changed")
    needs_update: bool = Field(description="Whether updates are needed")
    is_root_feature: bool = Field(
        default=False,
        description="Whether this is a root feature (no upstream dependencies)",
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
            elif self.is_root_feature:
                # Root features: can't determine update status without samples
                status_icon = "[blue]○[/blue]"
                status_text = "root feature"
            elif self.needs_update:
                status_icon = "[yellow]⚠[/yellow]"
                status_text = "needs update"
            else:
                status_icon = "[green]✓[/green]"
                status_text = "up-to-date"

        # For root features, don't show added/changed counts (they're not meaningful)
        if self.is_root_feature:
            return (
                f"{status_icon} {self.feature_key.to_string()} "
                f"(rows: {self.row_count}) — {status_text}"
            )

        return (
            f"{status_icon} {self.feature_key.to_string()} "
            f"(rows: {self.row_count}, added: {self.added_count}, "
            f"changed: {self.changed_count}) — {status_text}"
        )


class FeatureMetadataStatusWithIncrement(NamedTuple):
    """Feature metadata status paired with its LazyIncrement data.

    This combines a pure Pydantic status model with the LazyIncrement object
    needed for sample-level operations like generating previews.
    """

    status: FeatureMetadataStatus
    lazy_increment: LazyIncrement | None

    def _status_category(
        self,
    ) -> Literal["missing", "needs_update", "up_to_date", "root_feature"]:
        if not self.status.metadata_exists:
            return "missing"
        if self.status.is_root_feature:
            return "root_feature"
        if self.status.needs_update:
            return "needs_update"
        return "up_to_date"

    def sample_details(
        self,
        feature_cls: type[BaseFeature],
        *,
        limit: int = 5,
    ) -> list[str]:
        """Return formatted sample preview lines for verbose output."""
        if self.lazy_increment is None:
            return []

        id_columns_spec = feature_cls.spec().id_columns  # type: ignore[attr-defined]
        id_columns_seq = tuple(id_columns_spec) if id_columns_spec is not None else None

        return [
            line.strip()
            for line in format_sample_previews(
                self.lazy_increment,
                self.status.added_count,
                self.status.changed_count,
                id_columns_seq,
                limit=limit,
            )
        ]

    def to_representation(
        self,
        feature_cls: type[BaseFeature],
        *,
        verbose: bool,
    ) -> FullFeatureMetadataRepresentation:
        """Convert status to the full JSON representation used by the CLI."""
        sample_details = (
            self.sample_details(feature_cls)
            if verbose and self.lazy_increment
            else None
        )
        # For root features, added/changed are not meaningful
        added = None if self.status.is_root_feature else self.status.added_count
        changed = None if self.status.is_root_feature else self.status.changed_count

        return FullFeatureMetadataRepresentation(
            feature_key=self.status.feature_key.to_string(),
            status=self._status_category(),
            needs_update=self.status.needs_update,
            metadata_exists=self.status.metadata_exists,
            rows=self.status.row_count,
            added=added,
            changed=changed,
            target_version=self.status.target_version,
            is_root_feature=self.status.is_root_feature,
            sample_details=sample_details,
        )


def format_sample_previews(
    lazy_increment: LazyIncrement,
    added_count: int,
    changed_count: int,
    id_columns: Sequence[str] | None = None,
    limit: int = 5,
) -> list[str]:
    """Format sample previews for added and changed samples.

    Args:
        lazy_increment: The LazyIncrement containing added/changed samples
        added_count: Number of added samples (to avoid re-counting)
        changed_count: Number of changed samples (to avoid re-counting)
        id_columns: Columns to include in previews (defaults to ["sample_uid"])
        limit: Maximum number of samples to preview per category

    Returns:
        List of formatted preview lines
    """
    lines: list[str] = []
    cols = list(id_columns or ["sample_uid"])

    if added_count > 0:
        added_preview_df = (
            lazy_increment.added.select(cols).head(limit).collect().to_polars()
        )
        if added_preview_df.height > 0:
            preview_lines = [
                ", ".join(f"{col}={row[col]}" for col in added_preview_df.columns)
                for row in added_preview_df.to_dicts()
            ]
            lines.append("    Added samples: " + "; ".join(preview_lines))

    if changed_count > 0:
        changed_preview_df = (
            lazy_increment.changed.select(cols).head(limit).collect().to_polars()
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


def get_feature_metadata_status(
    feature_key: CoercibleToFeatureKey,
    metadata_store: MetadataStore,
    *,
    use_fallback: bool = False,
) -> FeatureMetadataStatusWithIncrement:
    """Get metadata status for a single feature.

    Args:
        feature_key: The feature key or feature class to check.
            Accepts a string ("a/b/c"), sequence of strings (["a", "b", "c"]),
            FeatureKey instance, or BaseFeature class.
        metadata_store: The metadata store to query
        use_fallback: Whether to read metadata row counts from fallback stores.
            When True, checks fallback stores if metadata is missing in the primary store.
            When False (default), only checks the primary store.
            Note: resolve_update always uses the primary store only.

    Returns:
        FeatureMetadataStatusWithIncrement containing status and lazy increment
    """
    from metaxy.metadata_store.exceptions import FeatureNotFoundError
    from metaxy.models.feature import FeatureGraph

    # Resolve to FeatureKey using the type adapter (handles all input types)
    key = ValidatedFeatureKeyAdapter.validate_python(feature_key)

    # Look up feature class from the active graph
    graph = FeatureGraph.get_active()
    if key not in graph.features_by_key:
        raise ValueError(f"Feature {key.to_string()} not found in active graph")
    feature_cls = graph.features_by_key[key]

    target_version = feature_cls.feature_version()

    # Check if this is a root feature (no upstream dependencies)
    plan = graph.get_feature_plan(key)
    is_root_feature = not plan.deps

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
        metadata_exists = True
    except FeatureNotFoundError:
        row_count = 0
        metadata_exists = False

    # For root features, we can't determine added/changed without samples
    if is_root_feature:
        status = FeatureMetadataStatus(
            feature_key=key,
            target_version=target_version,
            metadata_exists=metadata_exists,
            row_count=row_count,
            added_count=0,
            changed_count=0,
            needs_update=False,
            is_root_feature=True,
        )
        return FeatureMetadataStatusWithIncrement(status=status, lazy_increment=None)

    # For non-root features, resolve the update to get added/changed counts
    lazy_increment = metadata_store.resolve_update(
        feature_cls,
        lazy=True,
    )

    # Count changes
    added_count = count_lazy_rows(lazy_increment.added)
    changed_count = count_lazy_rows(lazy_increment.changed)

    status = FeatureMetadataStatus(
        feature_key=key,
        target_version=target_version,
        metadata_exists=metadata_exists,
        row_count=row_count,
        added_count=added_count,
        changed_count=changed_count,
        needs_update=added_count > 0 or changed_count > 0,
    )
    return FeatureMetadataStatusWithIncrement(
        status=status, lazy_increment=lazy_increment
    )
