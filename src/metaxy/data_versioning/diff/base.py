"""Abstract base class for metadata diff resolvers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    import polars as pl

TRef = TypeVar("TRef")  # Reference type for inputs (LazyFrame, ibis.Table, etc.)


@dataclass(frozen=True)
class DiffResult:
    """Result of diffing target versions with current metadata.

    Always contains materialized DataFrames (never None - empty DataFrames instead).
    This design ensures diff resolvers execute as few queries as possible:
    - SQL backends: One query with CTEs/UNION ALL computes all three, then splits
    - Polars: Collect once, split into three DataFrames

    Attributes:
        added: New samples [sample_id, data_version] (empty if none)
        changed: Changed samples [sample_id, data_version] (empty if none)
        removed: Removed samples [sample_id, data_version] (empty if none)
    """

    added: "pl.DataFrame"
    changed: "pl.DataFrame"
    removed: "pl.DataFrame"


class MetadataDiffResolver(ABC, Generic[TRef]):
    """Identifies rows with changed data_versions by comparing target with current.

    The diff resolver compares newly calculated data_versions (target) with
    existing metadata (current) to identify what needs to be written.

    This is Step 3 in the data versioning process:
    1. Join upstream features → unified upstream view
    2. Calculate data_version from upstream → target versions
    3. Diff with current metadata → identify changes ← THIS STEP

    Type Parameters:
        TRef: Input reference type (pl.LazyFrame for Polars, ibis.Table for SQL)

    Examples:
        - PolarsDiffResolver: Takes LazyFrames, uses anti-join, collects once
        - IbisDiffResolver: Takes ibis.Tables, uses SQL CTEs, executes once

    Important Design:
        Takes lazy/native refs as input, returns materialized DataFrames as output.
        This minimizes query execution:
        - SQL backends: One query with CTEs computes all three categories
        - Polars: Collects once, splits into three DataFrames

    Users can override Feature.resolve_data_version_diff to customize:
        - Ignore certain field changes
        - Apply custom change detection rules
        - Filter out specific samples
    """

    @abstractmethod
    def find_changes(
        self,
        target_versions: TRef,
        current_metadata: TRef | None,
    ) -> DiffResult:
        """Find all changes between target and current metadata.

        Compares target data_versions (newly calculated) with current metadata
        and categorizes all differences.

        Args:
            target_versions: Backend-specific ref with newly calculated data_versions
                (pl.LazyFrame for Polars, ibis.Table for SQL)
                Shape: [sample_id, data_version (calculated), upstream columns...]
            current_metadata: Backend-specific ref with current metadata, or None
                Shape: [sample_id, data_version (existing), feature_version, custom columns...]

        Returns:
            DiffResult with three materialized DataFrames (always pl.DataFrame):
            - added: New samples [sample_id, data_version] (empty if none)
            - changed: Changed samples [sample_id, data_version] (empty if none)
            - removed: Removed samples [sample_id, data_version] (empty if none)

        Implementation Note:
            Takes lazy/native refs as input, returns materialized DataFrames.
            Should execute as few queries as possible:
            - SQL backends: One query with CTEs for all three, then execute and split
            - Polars: Collect once, split into three DataFrames

        Note:
            For immutable append-only storage, typically only 'added' and 'changed'
            are written. 'removed' is useful for validation/reporting.
        """
        pass
