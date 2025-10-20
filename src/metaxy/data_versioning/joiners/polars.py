"""Polars implementation of upstream joiner."""

from typing import TYPE_CHECKING

import polars as pl

from metaxy.data_versioning.joiners.base import UpstreamJoiner

if TYPE_CHECKING:
    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.plan import FeaturePlan


class PolarsJoiner(UpstreamJoiner[pl.LazyFrame]):
    """Joins upstream features using Polars LazyFrames.

    Type Parameters:
        TRef = pl.LazyFrame

    Strategy:
    - Starts with first upstream feature
    - Sequentially inner joins remaining upstream features on sample_id
    - Renames data_version columns to avoid conflicts
    - All operations are lazy (no materialization until collect)
    """

    def join_upstream(
        self,
        upstream_refs: dict[str, pl.LazyFrame],
        feature_spec: "FeatureSpec",
        feature_plan: "FeaturePlan",
    ) -> tuple[pl.LazyFrame, dict[str, str]]:
        """Join upstream LazyFrames together.

        Args:
            upstream_refs: Dict of upstream feature key -> LazyFrame
            feature_spec: Feature specification
            feature_plan: Feature plan

        Returns:
            (joined LazyFrame, column mapping)
        """
        if not upstream_refs:
            # No upstream dependencies - source feature
            # Return empty LazyFrame with just sample_id column
            # The calculator will generate data_versions based only on feature_version
            return pl.LazyFrame({"sample_id": []}), {}

        # Start with the first upstream feature
        upstream_keys = sorted(upstream_refs.keys())
        first_key = upstream_keys[0]

        # Start with first upstream, rename its data_version column
        col_name_first = f"__upstream_{first_key}__data_version"
        joined = upstream_refs[first_key].select(
            ["sample_id", pl.col("data_version").alias(col_name_first)]
        )

        upstream_mapping = {first_key: col_name_first}

        # Join remaining upstream features
        for upstream_key in upstream_keys[1:]:
            col_name = f"__upstream_{upstream_key}__data_version"

            upstream_renamed = upstream_refs[upstream_key].select(
                ["sample_id", pl.col("data_version").alias(col_name)]
            )

            joined = joined.join(
                upstream_renamed,
                on="sample_id",
                how="inner",  # Only sample_ids present in ALL upstream
            )

            upstream_mapping[upstream_key] = col_name

        return joined, upstream_mapping
