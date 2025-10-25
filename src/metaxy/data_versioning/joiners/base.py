"""Abstract base class for upstream joiners."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import narwhals as nw

if TYPE_CHECKING:
    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.plan import FeaturePlan


class UpstreamJoiner(ABC):
    """Joins upstream feature metadata together.

    The joiner takes upstream feature metadata (which already has data_version columns)
    and joins them together to create a unified view of all dependencies.

    This is Step 1 in the data versioning process:
    1. Join upstream features → unified upstream view
    2. Calculate data_version from upstream → target versions
    3. Diff with current metadata → identify changes

    All component boundaries use Narwhals LazyFrames for backend-agnostic processing.

    Examples:
        - NarwhalsJoiner: Backend-agnostic using Narwhals expressions
        - IbisJoiner: Converts to Ibis internally for SQL processing
    """

    @abstractmethod
    def join_upstream(
        self,
        upstream_refs: dict[str, nw.LazyFrame[Any]],
        feature_spec: "FeatureSpec",
        feature_plan: "FeaturePlan",
        upstream_columns: dict[str, tuple[str, ...] | None] | None = None,
        upstream_renames: dict[str, dict[str, str] | None] | None = None,
    ) -> tuple[nw.LazyFrame[Any], dict[str, str]]:
        """Join all upstream features together with optional column selection/renaming.

        Joins upstream feature metadata on sample_uid to create a unified reference
        containing all upstream data_version columns needed for hash calculation,
        plus any additional user-specified columns.

        Args:
            upstream_refs: Upstream feature metadata Narwhals LazyFrames
                Keys are upstream feature keys (using to_string() format)
                Values are Narwhals LazyFrames with upstream metadata
            feature_spec: Specification of the feature being computed
            feature_plan: Resolved feature plan with dependencies
            upstream_columns: Optional dict mapping upstream feature keys to tuple of columns to keep.
                None or missing key = keep all columns. Empty tuple = only system columns.
            upstream_renames: Optional dict mapping upstream feature keys to rename dicts.
                Applied after column selection.

        Returns:
            Tuple of (joined_ref, upstream_column_mapping):
            - joined_ref: Narwhals LazyFrame with all upstream data joined
                Contains: sample_uid, data_version columns, and any user columns
            - upstream_column_mapping: Maps upstream feature key -> data_version column name
                Example: {"video": "__upstream_video__data_version"}

        Note:
            - Uses INNER join by default - only sample_uids present in ALL upstream features
              are included. This ensures we can compute valid data_versions.
            - System columns (sample_uid, data_version) are always preserved
            - User columns are preserved based on columns parameter (default: all)
        """
        pass
