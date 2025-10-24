"""Narwhals implementation of upstream joiner.

Unified joiner that works with any backend (Polars, Ibis/SQL) through Narwhals.
"""

from typing import TYPE_CHECKING

import narwhals as nw

from metaxy.data_versioning.joiners.base import UpstreamJoiner

if TYPE_CHECKING:
    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.plan import FeaturePlan


class NarwhalsJoiner(UpstreamJoiner):
    """Joins upstream features using Narwhals LazyFrames.

    Type Parameters:
        TRef = nw.LazyFrame (works with Polars, Ibis, Pandas, PyArrow)

    Strategy:
    - Starts with first upstream feature
    - Sequentially inner joins remaining upstream features on sample_id
    - Renames data_version columns to avoid conflicts
    - All operations are lazy (no materialization until collect)
    - Backend-agnostic: same code works for in-memory and SQL backends

    The underlying backend (Polars vs Ibis) is determined by what's wrapped:
    - nw.from_native(pl.LazyFrame) → stays in Polars
    - nw.from_native(ibis.Table) → stays in SQL until collect()
    """

    def join_upstream(
        self,
        upstream_refs: dict[str, nw.LazyFrame],
        feature_spec: "FeatureSpec",
        feature_plan: "FeaturePlan",
    ) -> tuple[nw.LazyFrame, dict[str, str]]:
        """Join upstream Narwhals LazyFrames together.

        Args:
            upstream_refs: Dict of upstream feature key -> Narwhals LazyFrame
            feature_spec: Feature specification
            feature_plan: Feature plan

        Returns:
            (joined Narwhals LazyFrame, column mapping)
        """
        if not upstream_refs:
            # No upstream dependencies - source feature
            # Return empty LazyFrame with just sample_id column (with proper type)
            import polars as pl

            # Create empty frame with explicit Int64 type for sample_id
            # This ensures it's not NULL-typed which would fail with Ibis backends
            empty_df = pl.LazyFrame(
                {"sample_id": pl.Series("sample_id", [], dtype=pl.Int64)}
            )
            return nw.from_native(empty_df), {}

        # Start with the first upstream feature
        upstream_keys = sorted(upstream_refs.keys())
        first_key = upstream_keys[0]

        # Start with first upstream, rename its data_version column
        col_name_first = f"__upstream_{first_key}__data_version"
        joined = upstream_refs[first_key].select(
            "sample_id", nw.col("data_version").alias(col_name_first)
        )

        upstream_mapping = {first_key: col_name_first}

        # Join remaining upstream features
        for upstream_key in upstream_keys[1:]:
            col_name = f"__upstream_{upstream_key}__data_version"

            upstream_renamed = upstream_refs[upstream_key].select(
                "sample_id", nw.col("data_version").alias(col_name)
            )

            joined = joined.join(
                upstream_renamed,
                on="sample_id",
                how="inner",  # Only sample_ids present in ALL upstream
            )

            upstream_mapping[upstream_key] = col_name

        return joined, upstream_mapping
