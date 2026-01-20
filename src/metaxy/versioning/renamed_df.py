from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generic

import narwhals as nw
from narwhals.typing import FrameT

if TYPE_CHECKING:
    from metaxy.versioning.feature_dep_transformer import IdColumnTracker


@dataclass(frozen=True)
class RenamedDataFrame(Generic[FrameT]):
    """An immutable wrapper for a dataframe with tracked ID columns.

    Each transformation method returns a new instance, preserving immutability.
    ID columns are tracked alongside the DataFrame for joining later.

    The id_column_tracker provides a complete picture of ID columns at each stage:
    - original: From upstream feature spec
    - renamed: After FeatureDep.rename
    - selected: After FeatureDep.columns selection
    - output: After lineage transformation (may differ for aggregation)
    """

    df: FrameT
    id_column_tracker: IdColumnTracker

    @property
    def id_columns(self) -> tuple[str, ...]:
        """ID columns for joining (the output ID columns after lineage transform).

        This is a convenience property that returns the effective ID columns
        to use when joining with other dependencies.
        """
        return self.id_column_tracker.output

    def with_id_tracker(self, id_column_tracker: IdColumnTracker) -> RenamedDataFrame[FrameT]:
        """Return a new RenamedDataFrame with an updated ID column tracker.

        Used by lineage handlers to update output ID columns after transformation.

        Args:
            id_column_tracker: The new ID column tracker

        Returns:
            New RenamedDataFrame with updated tracker.
        """
        return replace(self, id_column_tracker=id_column_tracker)

    def rename(self, mapping: Mapping[str, str]) -> RenamedDataFrame[FrameT]:
        """Rename columns and update ID tracking accordingly."""
        new_df = self.df.rename(mapping) if mapping else self.df  # ty: ignore[invalid-argument-type]

        # Update the tracker with renamed columns
        from metaxy.versioning.feature_dep_transformer import IdColumnTracker

        new_renamed = tuple(mapping.get(col, col) for col in self.id_column_tracker.original)
        # After rename, selected and output are same as renamed until selection
        new_tracker = IdColumnTracker(
            original=self.id_column_tracker.original,
            renamed=new_renamed,
            selected=new_renamed,  # Before selection, all renamed cols are "selected"
            output=new_renamed,  # Before lineage, output equals selected
        )

        return replace(self, df=new_df, id_column_tracker=new_tracker)

    def filter(self, filters: Sequence[nw.Expr] | None) -> RenamedDataFrame[FrameT]:
        """Filter rows (ID column tracking unchanged)."""
        if filters:
            new_df = self.df.filter(*filters)  # ty: ignore[invalid-argument-type]
            return replace(self, df=new_df)
        return self

    def select(self, columns: Sequence[str] | None) -> RenamedDataFrame[FrameT]:
        """Select columns and update ID tracking for selection."""
        if columns:
            new_df = self.df.select(*columns)  # ty: ignore[invalid-argument-type]

            # Update selected ID columns based on what's in the selection
            columns_set = set(columns)
            new_selected = tuple(col for col in self.id_column_tracker.renamed if col in columns_set)

            from metaxy.versioning.feature_dep_transformer import IdColumnTracker

            new_tracker = IdColumnTracker(
                original=self.id_column_tracker.original,
                renamed=self.id_column_tracker.renamed,
                selected=new_selected,
                output=new_selected,  # Before lineage, output equals selected
            )

            return replace(self, df=new_df, id_column_tracker=new_tracker)
        return self
