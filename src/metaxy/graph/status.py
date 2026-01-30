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
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.versioning.types import LazyIncrement


class FullFeatureMetadataRepresentation(BaseModel):
    """Full JSON-safe representation of feature metadata status."""

    feature_key: str
    status: Literal["missing", "needs_update", "up_to_date", "root_feature", "error"]
    needs_update: bool
    metadata_exists: bool
    store_rows: int
    missing: int | None
    stale: int | None
    orphaned: int | None
    target_version: str
    is_root_feature: bool = False
    sample_details: list[str] | None = None
    store_metadata: dict[str, Any] | None = None
    error_message: str | None = None
    progress_percentage: float | None = None


StatusCategory = Literal["missing", "needs_update", "up_to_date", "root_feature", "error"]

# Status display configuration
_STATUS_ICONS: dict[StatusCategory, str] = {
    "missing": "[red]✗[/red]",
    "root_feature": "[blue]○[/blue]",
    "needs_update": "[yellow]⚠[/yellow]",
    "up_to_date": "[green]✓[/green]",
    "error": "[red]![/red]",
}

_STATUS_TEXTS: dict[StatusCategory, str] = {
    "missing": "missing metadata",
    "root_feature": "root feature",
    "needs_update": "needs update",
    "up_to_date": "up-to-date",
    "error": "error",
}


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
    store_row_count: int = Field(description="Number of metadata rows currently in store (0 if none exist)")
    missing_count: int = Field(description="Number of new samples from upstream not yet in metadata")
    stale_count: int = Field(description="Number of samples with stale provenance needing update")
    orphaned_count: int = Field(description="Number of samples in store but removed from upstream")
    needs_update: bool = Field(description="Whether updates are needed")
    is_root_feature: bool = Field(
        default=False,
        description="Whether this is a root feature (no upstream dependencies)",
    )
    store_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Store-specific metadata (e.g., table_name, uri)",
    )
    progress_percentage: float | None = Field(
        default=None,
        description="Percentage of input units processed (0-100). None if not computed or root feature.",
    )

    @property
    def status_category(self) -> StatusCategory:
        """Compute the status category from current state."""
        if not self.metadata_exists:
            return "missing"
        if self.is_root_feature:
            return "root_feature"
        if self.needs_update:
            return "needs_update"
        return "up_to_date"

    def format_status_line(self) -> str:
        """Format a status line for display with Rich markup."""
        category = self.status_category
        icon = _STATUS_ICONS[category]
        text = _STATUS_TEXTS[category]
        key = self.feature_key.to_string()

        # Root features: don't show incoming/outdated/orphaned counts (not meaningful)
        if self.is_root_feature:
            return f"{icon} {key} (store: {self.store_row_count}) — {text}"

        return (
            f"{icon} {key} "
            f"(store: {self.store_row_count}, missing: {self.missing_count}, "
            f"stale: {self.stale_count}, orphaned: {self.orphaned_count}) — {text}"
        )


class FeatureMetadataStatusWithIncrement(NamedTuple):
    """Feature metadata status paired with its LazyIncrement data.

    This combines a pure Pydantic status model with the LazyIncrement object
    needed for sample-level operations like generating previews.
    """

    status: FeatureMetadataStatus
    lazy_increment: LazyIncrement | None

    @property
    def status_category(self) -> StatusCategory:
        """Delegate to the status model's category."""
        return self.status.status_category

    def sample_details(
        self,
        *,
        limit: int = 5,
    ) -> list[str]:
        """Return formatted sample preview lines for verbose output."""
        if self.lazy_increment is None:
            return []

        return [
            line.strip()
            for line in format_sample_previews(
                self.lazy_increment,
                self.status.missing_count,
                self.status.stale_count,
                self.status.orphaned_count,
                limit=limit,
            )
        ]

    def to_representation(
        self,
        *,
        verbose: bool,
    ) -> FullFeatureMetadataRepresentation:
        """Convert status to the full JSON representation used by the CLI."""
        sample_details = self.sample_details() if verbose and self.lazy_increment else None
        # For root features, missing/stale/orphaned are not meaningful
        missing = None if self.status.is_root_feature else self.status.missing_count
        stale = None if self.status.is_root_feature else self.status.stale_count
        orphaned = None if self.status.is_root_feature else self.status.orphaned_count

        return FullFeatureMetadataRepresentation(
            feature_key=self.status.feature_key.to_string(),
            status=self.status_category,
            needs_update=self.status.needs_update,
            metadata_exists=self.status.metadata_exists,
            store_rows=self.status.store_row_count,
            missing=missing,
            stale=stale,
            orphaned=orphaned,
            target_version=self.status.target_version,
            is_root_feature=self.status.is_root_feature,
            sample_details=sample_details,
            store_metadata=self.status.store_metadata or None,
            progress_percentage=self.status.progress_percentage,
        )


def format_sample_previews(
    lazy_increment: LazyIncrement,
    missing_count: int,
    stale_count: int,
    orphaned_count: int,
    limit: int = 5,
) -> list[str]:
    """Format sample previews for missing, stale, and orphaned samples.

    Args:
        lazy_increment: The LazyIncrement containing sample data
        missing_count: Number of missing samples (new from upstream)
        stale_count: Number of stale samples (outdated provenance)
        orphaned_count: Number of orphaned samples (removed from upstream)
        limit: Maximum number of samples to preview per category

    Returns:
        List of formatted preview lines with Rich markup
    """
    lines: list[str] = []

    if missing_count > 0:
        missing_preview_df = lazy_increment.added.head(limit).collect().to_polars()
        if missing_preview_df.height > 0:
            lines.append("[bold yellow]Missing samples:[/bold yellow]")
            glimpse_str = missing_preview_df.glimpse(return_type="string")
            if glimpse_str:
                lines.append(glimpse_str)

    if stale_count > 0:
        stale_preview_df = lazy_increment.changed.head(limit).collect().to_polars()
        if stale_preview_df.height > 0:
            lines.append("[bold cyan]Stale samples:[/bold cyan]")
            glimpse_str = stale_preview_df.glimpse(return_type="string")
            if glimpse_str:
                lines.append(glimpse_str)

    if orphaned_count > 0:
        orphaned_preview_df = lazy_increment.removed.head(limit).collect().to_polars()
        if orphaned_preview_df.height > 0:
            lines.append("[bold red]Orphaned samples:[/bold red]")
            glimpse_str = orphaned_preview_df.glimpse(return_type="string")
            if glimpse_str:
                lines.append(glimpse_str)

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
    use_fallback: bool = True,
    global_filters: Sequence[nw.Expr] | None = None,
    target_filters: Sequence[nw.Expr] | None = None,
    compute_progress: bool = False,
) -> FeatureMetadataStatusWithIncrement:
    """Get metadata status for a single feature.

    Args:
        feature_key: The feature key or feature class to check.
            Accepts a string ("a/b/c"), sequence of strings (["a", "b", "c"]),
            FeatureKey instance, or BaseFeature class.
        metadata_store: The metadata store to query
        use_fallback: Whether to read metadata from fallback stores.
        global_filters: List of Narwhals filter expressions applied to all features
            (both upstream and target).
        target_filters: List of Narwhals filter expressions applied only to the target
            feature (or more precisely, the result of an increment calculation on it).
        compute_progress: Whether to calculate progress percentage.
            When True, computes what percentage of input units have been processed.
            This requires additional computation (re-runs the input query).
            Default is False.

    Returns:
        FeatureMetadataStatusWithIncrement containing status and lazy increment
    """
    from metaxy.metadata_store.exceptions import FeatureNotFoundError
    from metaxy.models.feature import FeatureGraph

    # Resolve to FeatureKey using the type adapter (handles all input types)
    key = ValidatedFeatureKeyAdapter.validate_python(feature_key)

    # Look up feature definition from the active graph
    graph = FeatureGraph.get_active()
    if key not in graph.feature_definitions_by_key:
        raise ValueError(f"Feature {key.to_string()} not found in active graph")
    definition = graph.feature_definitions_by_key[key]

    target_version = graph.get_feature_version(key)

    # Check if this is a root feature (no upstream dependencies)
    plan = graph.get_feature_plan(key)
    is_root_feature = not plan.deps

    # Get row count for this feature version
    id_columns = definition.spec.id_columns
    id_columns_seq = tuple(id_columns) if id_columns is not None else None

    # Get store metadata (table_name, uri, etc.)
    store_metadata = metadata_store.get_store_metadata(key)

    # Combine global_filters and target_filters for read_metadata
    # (read_metadata doesn't distinguish them - it only reads the target feature)
    combined_filters: list[nw.Expr] = []
    if global_filters:
        combined_filters.extend(global_filters)
    if target_filters:
        combined_filters.extend(target_filters)

    try:
        metadata_lazy = metadata_store.read_metadata(
            key,
            columns=list(id_columns_seq) if id_columns_seq is not None else None,
            allow_fallback=use_fallback,
            filters=combined_filters if combined_filters else None,
        )
        row_count = count_lazy_rows(metadata_lazy)
        metadata_exists = True
    except FeatureNotFoundError:
        row_count = 0
        metadata_exists = False

    # For root features, we can't determine missing/stale/orphaned without samples
    if is_root_feature:
        status = FeatureMetadataStatus(
            feature_key=key,
            target_version=target_version,
            metadata_exists=metadata_exists,
            store_row_count=row_count,
            missing_count=0,
            stale_count=0,
            orphaned_count=0,
            needs_update=False,
            is_root_feature=True,
            store_metadata=store_metadata,
        )
        return FeatureMetadataStatusWithIncrement(status=status, lazy_increment=None)

    # For non-root features, resolve the update to get missing/stale/orphaned counts
    lazy_increment = metadata_store.resolve_update(
        key,
        lazy=True,
        global_filters=list(global_filters) if global_filters else None,
        target_filters=list(target_filters) if target_filters else None,
    )

    # Count changes
    missing_count = count_lazy_rows(lazy_increment.added)
    stale_count = count_lazy_rows(lazy_increment.changed)
    orphaned_count = count_lazy_rows(lazy_increment.removed)

    # Calculate progress if requested
    progress_percentage: float | None = None
    if compute_progress:
        progress_percentage = metadata_store.calculate_input_progress(lazy_increment, key)

    status = FeatureMetadataStatus(
        feature_key=key,
        target_version=target_version,
        metadata_exists=metadata_exists,
        store_row_count=row_count,
        missing_count=missing_count,
        stale_count=stale_count,
        orphaned_count=orphaned_count,
        needs_update=missing_count > 0 or stale_count > 0 or orphaned_count > 0,
        store_metadata=store_metadata,
        progress_percentage=progress_percentage,
    )
    return FeatureMetadataStatusWithIncrement(status=status, lazy_increment=lazy_increment)
