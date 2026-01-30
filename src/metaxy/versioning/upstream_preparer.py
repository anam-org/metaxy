"""Upstream data preparation for versioning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Generic

import narwhals as nw
from narwhals.typing import FrameT

from metaxy.models.constants import (
    METAXY_CREATED_AT,
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_DELETED_AT,
    METAXY_FEATURE_VERSION,
    METAXY_MATERIALIZATION_ID,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
    METAXY_SNAPSHOT_VERSION,
)
from metaxy.models.types import FeatureKey
from metaxy.versioning.renamed_df import RenamedDataFrame
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    from metaxy.models.plan import FeaturePlan
    from metaxy.versioning.engine import VersioningEngine


class UpstreamPreparer(Generic[FrameT]):
    """Prepares upstream DataFrames for provenance computation.

    Orchestrates the full upstream preparation pipeline:
    1. Transform each dependency (filter, rename, select columns)
    2. Apply lineage transformations (e.g., aggregate for N:1 relationships)
    3. Drop unnecessary system columns
    4. Validate no column collisions across features
    5. Join all dependencies into a single DataFrame
    """

    def __init__(self, plan: FeaturePlan, engine: VersioningEngine[FrameT]):
        self.plan = plan
        self.engine = engine

    def prepare(
        self,
        upstream: Mapping[FeatureKey, FrameT],
        filters: Mapping[FeatureKey, Sequence[nw.Expr]] | None,
        hash_algorithm: HashAlgorithm | None,
    ) -> FrameT:
        """Prepare and join upstream DataFrames for provenance computation.

        Runs the full preparation pipeline: transform, apply lineage, drop system
        columns, validate, and join all upstream features.

        Args:
            upstream: Dictionary mapping feature keys to their raw DataFrames.
            filters: Optional runtime filters to apply per feature.
            hash_algorithm: Required for aggregation lineage transformations.

        Returns:
            Single joined DataFrame ready for provenance computation.

        Raises:
            ValueError: If column collisions are detected across features.
        """
        assert len(upstream) > 0, "No upstream dataframes provided"

        # Step 1: Transform each dependency (filter, rename, select)
        dfs = self._transform_all(upstream, filters)

        # Step 2: Apply lineage transformations per-dependency (independently)
        if hash_algorithm is not None:
            dfs = self._apply_lineage_transforms(dfs, hash_algorithm)

        # Step 3: Drop system columns that aren't needed for provenance calculation
        dfs = self._drop_unnecessary_system_columns(dfs)

        # Step 4: Validate no column collisions (except ID columns and required system columns)
        if len(dfs) > 1:
            self._validate_no_collisions(dfs)

        # Step 5: Join all dependencies
        return self.engine.join(dfs)  # ty: ignore[invalid-argument-type]

    def _transform_all(
        self,
        upstream: Mapping[FeatureKey, FrameT],
        filters: Mapping[FeatureKey, Sequence[nw.Expr]] | None,
    ) -> dict[FeatureKey, RenamedDataFrame[FrameT]]:
        """Transform each upstream dependency using its FeatureDepTransformer."""
        return {
            k: self.engine.feature_transformers_by_key[k].transform(df, filters=(filters or {}).get(k))
            for k, df in upstream.items()
        }

    def _apply_lineage_transforms(
        self,
        dfs: dict[FeatureKey, RenamedDataFrame[FrameT]],
        hash_algorithm: HashAlgorithm,
    ) -> dict[FeatureKey, RenamedDataFrame[FrameT]]:
        """Apply lineage transformations per-dependency."""
        from metaxy.versioning.lineage_handler import create_lineage_handler

        result = dict(dfs)

        for feature_key, renamed_df in dfs.items():
            dep = self.plan.feature.deps_by_key.get(feature_key)
            if dep is not None:
                dep_transformer = self.engine.feature_transformers_by_key[feature_key]
                handler = create_lineage_handler(dep, self.plan, self.engine, dep_transformer)
                transformed_df = handler.transform_upstream(
                    renamed_df.df,  # ty: ignore[invalid-argument-type]
                    hash_algorithm,
                )
                result[feature_key] = RenamedDataFrame(  # ty: ignore[invalid-assignment]
                    df=transformed_df,
                    id_column_tracker=renamed_df.id_column_tracker,
                )

        return result

    def _drop_unnecessary_system_columns(
        self,
        dfs: dict[FeatureKey, RenamedDataFrame[FrameT]],
    ) -> dict[FeatureKey, RenamedDataFrame[FrameT]]:
        """Drop system columns not needed for provenance calculation."""
        columns_to_drop = [
            METAXY_FEATURE_VERSION,
            METAXY_SNAPSHOT_VERSION,
        ]

        result = dict(dfs)

        for feature_key, renamed_df in dfs.items():
            cols = renamed_df.df.collect_schema().names()  # ty: ignore[invalid-argument-type]
            cols_to_drop = [col for col in columns_to_drop if col in cols]
            if cols_to_drop:
                result[feature_key] = RenamedDataFrame(  # ty: ignore[invalid-assignment]
                    df=renamed_df.df.drop(*cols_to_drop),  # ty: ignore[invalid-argument-type]
                    id_column_tracker=renamed_df.id_column_tracker,
                )

        return result

    def _validate_no_collisions(
        self,
        dfs: dict[FeatureKey, RenamedDataFrame[FrameT]],
    ) -> None:
        """Raise ValueError if non-ID, non-system columns appear in multiple features."""
        all_columns: dict[str, list[FeatureKey]] = {}
        for feature_key, renamed_df in dfs.items():
            cols = renamed_df.df.collect_schema().names()  # ty: ignore[invalid-argument-type]
            for col in cols:
                if col not in all_columns:
                    all_columns[col] = []
                all_columns[col].append(feature_key)

        # System columns that are allowed to collide (needed for provenance calculation)
        allowed_system_columns = {
            METAXY_PROVENANCE,
            METAXY_PROVENANCE_BY_FIELD,
            METAXY_DATA_VERSION,
            METAXY_DATA_VERSION_BY_FIELD,
            METAXY_CREATED_AT,
            METAXY_DELETED_AT,
            METAXY_MATERIALIZATION_ID,
        }
        id_cols = set(self.engine.shared_id_columns)
        colliding_columns = [
            col
            for col, features in all_columns.items()
            if len(features) > 1 and col not in id_cols and col not in allowed_system_columns
        ]

        if colliding_columns:
            raise ValueError(
                f"Found column collisions {colliding_columns} across upstream features for feature {self.plan.feature.key}: "
                f"Only ID columns {list(id_cols)} and required system columns {list(allowed_system_columns)} should be shared. "
                f"Please add explicit renames in your FeatureDep to avoid column collisions."
            )
