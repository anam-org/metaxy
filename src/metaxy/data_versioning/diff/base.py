"""Abstract base class for metadata diff resolvers."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import narwhals as nw

if TYPE_CHECKING:
    pass


class LazyDiffResult(NamedTuple):
    """Result of resolving an incremental update with lazy Narwhals LazyFrames.

    Users can:
    - Keep lazy for further operations: result.added.filter(...)

    - Materialize to Polars: result.added.collect().to_native()

    - Convert to DiffResult: result.collect()

    Attributes:
        added: New samples that appear upstream and haven't been processed yet.

            Columns: `[*user_defined_columns, "data_version"]`
        changed: Samples with new data versions that should be re-processed.

            Columns: `[*user_defined_columns, "data_version"]`
        removed: Samples that have been previously processed but have been removed from upstream since that.

            Columns: `[*id_columns, "data_version"]`

    Note:
        `added` and `changed` contain all the user-defined columns, but `removed` only contains the ID columns.
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
    """Result of resolving an incremental update with eager Narwhals DataFrames.

    Contains materialized Narwhals DataFrames.

    Users can convert to their preferred format:
    - Polars: result.added.to_native()

    Attributes:
        added: New samples that appear upstream and haven't been processed yet.

            Columns: `[*user_defined_columns, "data_version"]`
        changed: Samples with new data versions that should be re-processed.

            Columns: `[*user_defined_columns, "data_version"]`
        removed: Samples that have been previously processed but have been removed from upstream since that.

            Columns: `[*id_columns, "data_version"]`

    Note:
        `added` and `changed` contain all the user-defined columns, but `removed` only contains the ID columns.
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
        id_columns: Sequence[str],
    ) -> LazyDiffResult:
        """Find all changes between target and current metadata.

        Compares target data_versions (newly calculated) with current metadata
        and categorizes all differences.

        Args:
            target_versions: Narwhals LazyFrame with newly calculated data_versions
                Shape: [ID columns, data_version (calculated), upstream columns...]
            current_metadata: Narwhals LazyFrame with current metadata, or None
                Shape: [ID columns, data_version (existing), feature_version, custom columns...]
                Should be pre-filtered by feature_version at the caller level if needed.
            id_columns: List of ID columns to use for comparison (required - from feature spec)

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

            ID columns must be explicitly provided from the feature spec.
        """
        pass
