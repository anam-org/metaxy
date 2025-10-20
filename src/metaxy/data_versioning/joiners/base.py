"""Abstract base class for upstream joiners."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

TRef = TypeVar("TRef")  # Reference type (LazyFrame, ibis.Table, etc.)

if TYPE_CHECKING:
    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.plan import FeaturePlan


class UpstreamJoiner(ABC, Generic[TRef]):
    """Joins upstream feature metadata together.

    The joiner takes upstream feature metadata (which already has data_version columns)
    and joins them together to create a unified view of all dependencies.

    This is Step 1 in the data versioning process:
    1. Join upstream features → unified upstream view
    2. Calculate data_version from upstream → target versions
    3. Diff with current metadata → identify changes

    Type Parameters:
        TRef: Backend-specific reference type (pl.LazyFrame, ibis.Table, etc.)

    Examples:
        - PolarsJoiner: Joins LazyFrames using Polars operations
        - IbisJoiner: Joins Ibis tables using SQL
        - HybridJoiner: Uploads to temp tables then SQL joins
    """

    @abstractmethod
    def join_upstream(
        self,
        upstream_refs: dict[str, TRef],
        feature_spec: "FeatureSpec",
        feature_plan: "FeaturePlan",
    ) -> tuple[TRef, dict[str, str]]:
        """Join all upstream features together.

        Joins upstream feature metadata on sample_id to create a unified reference
        containing all upstream data_version columns needed for hash calculation.

        Args:
            upstream_refs: Upstream feature metadata references
                Keys are upstream feature keys (using to_string() format)
                Values are backend-specific references to upstream metadata
            feature_spec: Specification of the feature being computed
            feature_plan: Resolved feature plan with dependencies

        Returns:
            Tuple of (joined_ref, upstream_column_mapping):
            - joined_ref: Reference with all upstream data joined
                Shape: [sample_id, __upstream_video__data_version, __upstream_audio__data_version, ...]
            - upstream_column_mapping: Maps upstream feature key -> column name
                Example: {"video": "__upstream_video__data_version"}

        Note:
            Uses INNER join by default - only sample_ids present in ALL upstream features
            are included. This ensures we can compute valid data_versions.
        """
        pass
