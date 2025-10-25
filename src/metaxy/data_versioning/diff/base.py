"""Abstract base class for metadata diff resolvers."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, NamedTuple

import narwhals as nw

if TYPE_CHECKING:
    pass


class LazyDiffResult(NamedTuple):
    """Result of diffing with lazy Narwhals LazyFrames (opt-in via lazy=True).

    Contains lazy Narwhals LazyFrames - users decide when/how to materialize.

    Users can:
    - Keep lazy for further operations: result.added.filter(...)
    - Materialize to Polars: result.added.collect().to_native()
    - Materialize to Pandas: result.added.collect().to_pandas()
    - Materialize to PyArrow: result.added.collect().to_arrow()
    - Convert to DiffResult: result.collect()

    Backend execution:
    - SQL stores: All operations stay in SQL until .collect()
    - Polars stores: Operations stay lazy until .collect()

    Attributes:
        added: New samples (lazy, never None - empty LazyFrame instead)
            Columns: [sample_uid, data_version, ...user columns...]
        changed: Changed samples (lazy, never None)
            Columns: [sample_uid, data_version, ...user columns...]
        removed: Removed samples (lazy, never None)
            Columns: [sample_uid, data_version, ...user columns...]

    Note:
        May contain additional user columns beyond sample_uid and data_version,
        depending on what was passed to resolve_update() via align_upstream_metadata.
    """

    added: nw.LazyFrame[Any]
    changed: nw.LazyFrame[Any]
    removed: nw.LazyFrame[Any]

    def collect(self) -> "DiffResult":
        """Materialize all lazy frames to create a DiffResult.

        Returns:
            DiffResult with all frames materialized to eager DataFrames.
        """
        return DiffResult(
            added=self.added.collect(),
            changed=self.changed.collect(),
            removed=self.removed.collect(),
        )


class DiffResult(NamedTuple):
    """Result of diffing with eager Narwhals DataFrames (default).

    Contains materialized Narwhals DataFrames - ready to use immediately.

    Users can convert to their preferred format:
    - Polars: result.added.to_native()
    - Pandas: result.added.to_pandas()
    - PyArrow: result.added.to_arrow()

    Attributes:
        added: New samples (eager, never None - empty DataFrame instead)
            Columns: [sample_uid, data_version, ...user columns...]
        changed: Changed samples (eager, never None)
            Columns: [sample_uid, data_version, ...user columns...]
        removed: Removed samples (eager, never None)
            Columns: [sample_uid, data_version, ...user columns...]

    Note:
        May contain additional user columns beyond sample_uid and data_version,
        depending on what was passed to resolve_update() via align_upstream_metadata.
    """

    added: nw.DataFrame[Any]
    changed: nw.DataFrame[Any]
    removed: nw.DataFrame[Any]


class MetadataDiffResolver(ABC):
    """Identifies rows with changed data_versions by comparing target with current.

    The diff resolver compares newly calculated data_versions (target) with
    existing metadata (current) to identify what needs to be written.

    This is Step 3 in the data versioning process:
    1. Join upstream features → unified upstream view
    2. Calculate data_version from upstream → target versions
    3. Diff with current metadata → identify changes ← THIS STEP

    All component boundaries use Narwhals LazyFrames for backend-agnostic processing.

    Examples:
        - NarwhalsDiffResolver: Backend-agnostic using Narwhals expressions
        - IbisDiffResolver: Converts to Ibis internally for SQL processing

    Important Design:
        Takes lazy Narwhals refs as input, returns LazyDiffResult as output.
        This minimizes query execution:
        - SQL backends: One query with CTEs computes all three categories
        - Polars: Uses lazy operations, splits into three LazyFrames

    Users can override Feature.resolve_data_version_diff to customize:
        - Ignore certain field changes
        - Apply custom change detection rules
        - Filter out specific samples
    """

    @abstractmethod
    def find_changes(
        self,
        target_versions: nw.LazyFrame[Any],
        current_metadata: nw.LazyFrame[Any] | None,
    ) -> LazyDiffResult:
        """Find all changes between target and current metadata.

        Compares target data_versions (newly calculated) with current metadata
        and categorizes all differences.

        Args:
            target_versions: Narwhals LazyFrame with newly calculated data_versions
                Shape: [sample_uid, data_version (calculated), upstream columns...]
            current_metadata: Narwhals LazyFrame with current metadata, or None
                Shape: [sample_uid, data_version (existing), feature_version, custom columns...]
                Should be pre-filtered by feature_version at the caller level if needed.

        Returns:
            LazyDiffResult with three lazy Narwhals LazyFrames.
            Caller materializes to DiffResult if needed (for lazy=False).

        Implementation Note:
            Should build lazy operations without materializing:
            - SQL backends: Build one lazy query with CTEs for all three categories
            - Polars: Use lazy operations, no collect() calls

        Note:
            For immutable append-only storage, typically only 'added' and 'changed'
            are written. 'removed' is useful for validation/reporting.

            Feature version filtering should happen at the read_metadata() level,
            not in the diff resolver. The diff resolver just compares whatever
            metadata is passed to it.
        """
        pass
