"""Increment resolution for versioning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, cast

import narwhals as nw
from narwhals.typing import FrameT

from metaxy.models.constants import (
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
)

if TYPE_CHECKING:
    from metaxy.models.plan import FeaturePlan
    from metaxy.versioning.engine import VersioningEngine


class IncrementResolver(Generic[FrameT]):
    """Compares expected provenance with current metadata to identify changes.

    Used during incremental updates to determine which samples need to be
    added (new samples), changed (provenance differs), or removed (no longer
    in expected data). Handles lineage-specific transformations for proper
    comparison (e.g., collapsing expanded rows).
    """

    def __init__(self, plan: FeaturePlan, engine: VersioningEngine[FrameT]):
        self.plan = plan
        self.engine = engine

    def resolve(
        self,
        expected: FrameT,
        current: FrameT | None,
        join_columns: list[str],
    ) -> tuple[FrameT, FrameT | None, FrameT | None]:
        """Compare expected and current metadata to find incremental changes.

        Performs set operations using provenance columns to identify:
        - Added: Samples in expected but not in current (anti-join)
        - Changed: Samples in both with different provenance (inner join + filter)
        - Removed: Samples in current but not in expected (anti-join)

        Args:
            expected: DataFrame with expected provenance from upstream.
            current: Current metadata from the store, or None for initial load.
            join_columns: Columns to use for matching samples.

        Returns:
            Tuple of (added, changed, removed) DataFrames. Changed and removed
            are None if current is None.
        """
        if current is None:
            return expected, None, None

        # Apply per-dependency lineage handling to current data
        current = self._transform_current_for_comparison(current, join_columns)

        # Rename provenance columns in current to avoid conflicts
        current = current.rename(  # ty: ignore[invalid-argument-type]
            {
                METAXY_PROVENANCE: f"__current_{METAXY_PROVENANCE}",
                METAXY_PROVENANCE_BY_FIELD: f"__current_{METAXY_PROVENANCE_BY_FIELD}",
            }
        )

        # Find added samples (in expected but not current)
        added = cast(
            FrameT,
            expected.join(  # ty: ignore[invalid-argument-type]
                cast(FrameT, current.select(join_columns)),  # ty: ignore[invalid-argument-type]
                on=join_columns,
                how="anti",
            ),
        )

        # Find changed samples (in both but different provenance)
        changed = cast(
            FrameT,
            expected.join(  # ty: ignore[invalid-argument-type]
                cast(  # ty: ignore[invalid-argument-type]
                    FrameT,
                    current.select(*join_columns, f"__current_{METAXY_PROVENANCE}"),
                ),
                on=join_columns,
                how="inner",
            )
            .filter(
                nw.col(f"__current_{METAXY_PROVENANCE}").is_null()
                | (nw.col(METAXY_PROVENANCE) != nw.col(f"__current_{METAXY_PROVENANCE}"))
            )
            .drop(f"__current_{METAXY_PROVENANCE}"),
        )

        # Find removed samples (in current but not expected)
        removed = cast(
            FrameT,
            current.join(  # ty: ignore[invalid-argument-type]
                cast(FrameT, expected.select(join_columns)),  # ty: ignore[invalid-argument-type]
                on=join_columns,
                how="anti",
            ).rename(
                {
                    f"__current_{METAXY_PROVENANCE}": METAXY_PROVENANCE,
                    f"__current_{METAXY_PROVENANCE_BY_FIELD}": METAXY_PROVENANCE_BY_FIELD,
                }
            ),
        )

        return added, changed, removed

    def _transform_current_for_comparison(
        self,
        current: FrameT,
        join_columns: list[str],
    ) -> FrameT:
        """Apply per-dependency lineage handling to current data for comparison."""
        from metaxy.versioning.lineage_handler import create_lineage_handler

        for dep in self.plan.feature_deps or []:
            dep_transformer = self.engine.feature_transformers_by_key[dep.feature]
            lineage_handler = create_lineage_handler(dep, self.plan, self.engine, dep_transformer)
            current = lineage_handler.transform_current_for_comparison(
                current,  # ty: ignore[invalid-argument-type]
                join_columns,
            )

        return current
