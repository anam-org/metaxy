from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Literal, cast

import narwhals as nw
from narwhals.typing import FrameT

from metaxy._hashing import get_hash_truncation_length
from metaxy.config import MetaxyConfig
from metaxy.models.constants import (
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.plan import FeaturePlan, FQFieldKey
from metaxy.models.types import FeatureKey
from metaxy.versioning.feature_dep_transformer import FeatureDepTransformer
from metaxy.versioning.renamed_df import RenamedDataFrame
from metaxy.versioning.types import HashAlgorithm
from metaxy.versioning.validation import validate_column_configuration

# Lazy imports to avoid circular dependencies
if TYPE_CHECKING:
    from metaxy.versioning.increment_resolver import IncrementResolver
    from metaxy.versioning.upstream_preparer import UpstreamPreparer


class VersioningEngine(ABC):
    """Engine for computing and tracking sample and field level provenance.

    This is the core abstraction for versioning operations.
    Each backend implements this class to provide database-specific hashing
    and aggregation functions while sharing the provenance computation logic.

    The engine handles:

    - Joining upstream feature DataFrames with appropriate join types

    - Computing provenance hashes from upstream data versions

    - Resolving incremental changes (added, changed, removed samples)
    """

    def __init__(self, plan: FeaturePlan):
        validate_column_configuration(plan)
        self.plan = plan

    @classmethod
    @abstractmethod
    def implementation(cls) -> nw.Implementation: ...

    @cached_property
    def key(self) -> FeatureKey:
        """Feature key for the feature we are calculating provenance for."""
        return self.plan.feature.key

    @cached_property
    def feature_transformers_by_key(self) -> dict[FeatureKey, FeatureDepTransformer]:
        """Build transformers for each upstream dependency."""
        return {dep.feature: FeatureDepTransformer(dep=dep, plan=self.plan) for dep in (self.plan.feature_deps or [])}

    @cached_property
    def shared_id_columns(self) -> list[str]:
        """ID columns common across all upstream features for joining."""
        cols = self.plan.input_id_columns or list(self.plan.feature.id_columns)

        if not cols:
            raise ValueError(
                f"No shared ID columns found for upstream features of feature {self.key}. "
                f"Please ensure that there is at least one ID column shared across all upstream features. "
                f"Consider tweaking the `rename` field on the `FeatureDep` objects of {self.key} feature spec, "
                f"or check your lineage relationship configurations."
            )

        return cols

    def join(self, upstream: Mapping[FeatureKey, RenamedDataFrame[FrameT]]) -> FrameT:
        """Join renamed upstream DataFrames respecting optional/required dependencies.

        Required dependencies use inner joins, optional dependencies use left or full
        outer joins depending on whether any required dependencies exist. ID columns
        are coalesced for full outer joins to handle NULLs from either side.

        Args:
            upstream: Dictionary mapping feature keys to their renamed DataFrames.

        Returns:
            Single joined DataFrame containing columns from all upstream features.
        """
        assert len(upstream) > 0, "No upstream dataframes provided"

        # If no feature_deps, fall back to original behavior (all inner joins)
        if not self.plan.feature_deps:
            key, renamed_df = next(iter(upstream.items()))
            df = renamed_df.df
            for next_key, renamed_df in upstream.items():
                if key == next_key:
                    continue
                df = cast(
                    FrameT,
                    df.join(renamed_df.df, on=self.shared_id_columns, how="inner"),  # ty: ignore[invalid-argument-type]
                )
            return df

        # Use cached properties for required and optional dependencies
        required_deps = self.plan.required_deps
        optional_deps = self.plan.optional_deps

        # Check if all dependencies are optional
        all_optional = len(required_deps) == 0

        df: FrameT | None = None

        # First, join all required dependencies with inner join
        for dep in required_deps:
            if dep.feature not in upstream:
                continue

            renamed_df = upstream[dep.feature]

            if df is None:
                df = renamed_df.df
            else:
                df = cast(
                    FrameT,
                    df.join(renamed_df.df, on=self.shared_id_columns, how="inner"),  # ty: ignore[invalid-argument-type]
                )

        # Then, join all optional dependencies
        # Use outer join if all deps are optional, left join otherwise
        optional_join_type: Literal["left", "full"] = "full" if all_optional else "left"

        for dep in optional_deps:
            if dep.feature not in upstream:
                continue

            renamed_df = upstream[dep.feature]

            if df is None:
                # First dependency when all are optional - becomes the base
                df = renamed_df.df
            else:
                df = cast(
                    FrameT,
                    df.join(  # ty: ignore[invalid-argument-type]
                        renamed_df.df,  # ty: ignore[invalid-argument-type]
                        on=self.shared_id_columns,
                        how=optional_join_type,
                    ),
                )

                # For full outer join, coalesce the ID columns
                if optional_join_type == "full":
                    for id_col in self.shared_id_columns:
                        right_col = f"{id_col}_right"
                        # Check if the right column exists (ty ignore for generic type)
                        col_names = df.collect_schema().names()  # ty: ignore[invalid-argument-type]
                        if right_col in col_names:
                            df = cast(
                                FrameT,
                                df.with_columns(  # ty: ignore[invalid-argument-type]
                                    nw.coalesce(nw.col(id_col), nw.col(right_col)).alias(id_col)
                                ).drop(right_col),
                            )

        assert df is not None, "No dataframes were joined"
        return df

    def prepare_upstream(
        self,
        upstream: Mapping[FeatureKey, FrameT],
        filters: Mapping[FeatureKey, Sequence[nw.Expr]] | None,
        hash_algorithm: HashAlgorithm | None = None,
    ) -> FrameT:
        """Prepare and join upstream DataFrames for provenance computation.

        Applies filtering, renaming, column selection, and lineage transformations
        based on FeatureDep configuration, then joins all upstream features.

        Args:
            upstream: Dictionary mapping feature keys to their DataFrames.
            filters: Optional runtime filters to apply per feature.
            hash_algorithm: Required for aggregation lineage transformations.

        Returns:
            Joined DataFrame ready for provenance computation.
        """
        from metaxy.versioning.upstream_preparer import UpstreamPreparer

        preparer: UpstreamPreparer[FrameT] = UpstreamPreparer(self.plan, self)
        return preparer.prepare(upstream, filters, hash_algorithm)

    @abstractmethod
    def hash_string_column(
        self,
        df: FrameT,
        source_column: str,
        target_column: str,
        hash_algo: HashAlgorithm,
        truncate_length: int | None = None,
    ) -> FrameT:
        """Hash a string column using the backend-specific hash function.

        Args:
            df: Input DataFrame.
            source_column: Name of the string column to hash.
            target_column: Name for the new column containing the hash.
            hash_algo: Hash algorithm to use (e.g., MD5, SHA256).
            truncate_length: If provided, truncate the hash to this length.

        Returns:
            DataFrame with the new hashed column added.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def build_struct_column(
        df: FrameT,
        struct_name: str,
        field_columns: dict[str, str],
    ) -> FrameT:
        """Build a struct column from existing columns.

        Args:
            df: Input DataFrame.
            struct_name: Name for the new struct column.
            field_columns: Mapping of struct field names to source column names.

        Returns:
            DataFrame with the new struct column added.
        """
        raise NotImplementedError()

    @abstractmethod
    def concat_strings_over_groups(
        self,
        df: FrameT,
        source_column: str,
        target_column: str,
        group_by_columns: list[str],
        order_by_columns: list[str],
        separator: str = "|",
    ) -> FrameT:
        """Concatenate string values within groups using window functions.

        Used for aggregation lineage to combine provenance values from multiple
        rows into a single deterministic string for hashing.

        Args:
            df: Input DataFrame.
            source_column: Name of the string column to concatenate.
            target_column: Name for the new column containing concatenated values.
            group_by_columns: Columns defining the groups.
            order_by_columns: Columns for deterministic ordering within groups.
            separator: Separator between concatenated values.

        Returns:
            DataFrame with the new concatenated column added.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def keep_latest_by_group(
        df: FrameT,
        group_columns: list[str],
        timestamp_columns: list[str],
    ) -> FrameT:
        """Keep only the latest row per group based on timestamp columns.

        Args:
            df: Input DataFrame.
            group_columns: Columns defining the groups.
            timestamp_columns: Column names to coalesce for ordering (uses first non-null value).

        Returns:
            DataFrame with only the latest row per group.
        """
        raise NotImplementedError()

    def _extract_metadata_fields(self, df: FrameT, col_name: str, field_mapping: dict[str, str]) -> FrameT:
        """Extract fields from a metadata column and add as new columns.

        Args:
            df: Input DataFrame.
            col_name: Name of the metadata column (Struct or Map).
            field_mapping: Mapping from field names to output column names.
        """
        from metaxy.utils.dataframes import find_map_columns

        if col_name not in find_map_columns(df):
            exprs = [
                nw.col(col_name).struct.field(field_name).cast(nw.String).alias(out_col)
                for field_name, out_col in field_mapping.items()
            ]
            return df.with_columns(*exprs)  # ty: ignore[invalid-argument-type]

        # Map path: use native polars-map expressions (preserves laziness)
        import polars as pl
        import polars_map  # noqa: F401  # registers .map accessor

        native = df.to_native()  # ty: ignore[invalid-argument-type]
        exprs = [
            pl.col(col_name).map.get(field_name).cast(pl.String).alias(out_col)  # ty: ignore[unresolved-attribute]
            for field_name, out_col in field_mapping.items()
        ]
        return cast(FrameT, nw.from_native(native.with_columns(exprs)))

    def aggregate_metadata_columns(
        self,
        df: FrameT,
        group_columns: list[str],
        order_by_columns: list[str],
        upstream_field_names: list[str],
        renamed_data_version_col: str,
        renamed_data_version_by_field_col: str,
        renamed_prov_col: str,
        renamed_prov_by_field_col: str,
        hash_algorithm: HashAlgorithm,
    ) -> FrameT:
        """Aggregate metadata columns within groups for aggregation lineage.

        For N:1 aggregation relationships, this method combines provenance values
        from multiple upstream rows into a single hash per group. Uses window
        functions so all rows in the same group receive identical metadata values
        without reducing the number of rows.

        Args:
            df: Input DataFrame with upstream metadata columns.
            group_columns: Columns defining aggregation groups.
            order_by_columns: Columns for deterministic ordering within groups.
            upstream_field_names: Field names from the upstream feature spec.
            renamed_data_version_col: Renamed data_version column name.
            renamed_data_version_by_field_col: Renamed data_version_by_field column name.
            renamed_prov_col: Renamed provenance column name.
            renamed_prov_by_field_col: Renamed provenance_by_field column name.
            hash_algorithm: Hash algorithm for combining values.

        Returns:
            DataFrame with aggregated metadata columns (same row count as input).
        """
        # Step 1: Extract each field value from metadata column to a temp column
        extracted_cols: dict[str, str] = {fn: f"__extract_{fn}" for fn in upstream_field_names}
        df = self._extract_metadata_fields(df, renamed_data_version_by_field_col, extracted_cols)

        # Step 2: Use window function to aggregate within groups
        aggregated_cols: dict[str, str] = {}  # field_name -> aggregated_col_name
        for field_name, extracted_col in extracted_cols.items():
            aggregated_col = f"__agg_{field_name}"
            aggregated_cols[field_name] = aggregated_col

            df = self.concat_strings_over_groups(
                df,  # ty: ignore[invalid-argument-type]
                source_column=extracted_col,
                target_column=aggregated_col,
                group_by_columns=group_columns,
                order_by_columns=order_by_columns,
                separator="|",
            )

        # Step 3: Hash each aggregated field
        hash_length = get_hash_truncation_length()
        hashed_field_cols: dict[str, str] = {}  # field_name -> hashed_col_name

        for field_name, aggregated_col in aggregated_cols.items():
            hash_col = f"__hash_{field_name}"
            hashed_field_cols[field_name] = hash_col

            df = self.hash_string_column(
                df,  # ty: ignore[invalid-argument-type]
                aggregated_col,
                hash_col,
                hash_algorithm,
                truncate_length=hash_length,
            )

        # Drop the original struct columns and temp columns, then rebuild structs
        df = df.drop(  # ty: ignore[invalid-argument-type]
            renamed_data_version_by_field_col,
            renamed_prov_by_field_col,
            renamed_data_version_col,
            renamed_prov_col,
            *extracted_cols.values(),
            *aggregated_cols.values(),
        )

        # Build new struct columns from hashed fields
        df = self.build_struct_column(  # ty: ignore[invalid-assignment]
            df, renamed_data_version_by_field_col, hashed_field_cols
        )
        df = self.build_struct_column(
            df,
            renamed_prov_by_field_col,
            hashed_field_cols,
        )

        # Compute sample-level data_version and provenance by hashing all fields together
        # Pre-extract fields, then concatenate with separator and hash
        sorted_fields = sorted(upstream_field_names)
        temp_field_cols = {fn: f"__field_{fn}" for fn in sorted_fields}
        df = self._extract_metadata_fields(df, renamed_data_version_by_field_col, temp_field_cols)
        sample_concat = nw.concat_str([nw.col(c) for c in temp_field_cols.values()], separator="|")
        df = df.with_columns(sample_concat.alias("__sample_concat"))  # ty: ignore[invalid-argument-type]

        df = self.hash_string_column(  # ty: ignore[invalid-assignment]
            df,
            "__sample_concat",
            renamed_data_version_col,
            hash_algorithm,
            truncate_length=hash_length,
        )
        df = df.with_columns(  # ty: ignore[invalid-argument-type]
            nw.col(renamed_data_version_col).alias(renamed_prov_col),
        )

        # Drop temp columns
        df = df.drop("__sample_concat", *hashed_field_cols.values(), *temp_field_cols.values())

        return df

    def get_renamed_data_version_by_field_col(self, feature_key: FeatureKey) -> str:
        """Get the renamed data_version_by_field column name for an upstream feature."""
        return self.feature_transformers_by_key[feature_key].renamed_data_version_by_field_col

    def _compute_provenance_internal(
        self,
        df: FrameT,
        hash_algo: HashAlgorithm,
        drop_renamed_data_version_col: bool = False,
    ) -> FrameT:
        """Compute provenance columns from a DataFrame with upstream data_version_by_field columns."""
        hash_length = MetaxyConfig.get().hash_truncation_length

        # Build concatenation columns for each field
        temp_concat_cols: dict[str, str] = {}  # field_key_str -> temp_col_name

        # Pre-extract upstream data_version_by_field values as temporary columns
        pre_extracted_temp_cols: list[str] = []
        parent_field_col_map: dict[FQFieldKey, str] = {}  # fq_key -> temp_col_name

        for field_spec in self.plan.feature.fields:
            parent_fields = self.plan.get_parent_fields_for_field(field_spec.key)
            for fq_key, parent_field_spec in parent_fields.items():
                if fq_key in parent_field_col_map:
                    continue
                col_name = self.get_renamed_data_version_by_field_col(fq_key.feature)
                struct_key = parent_field_spec.key.to_struct_key()
                temp_col = f"__pf_{fq_key.feature.table_name}_{struct_key}"
                parent_field_col_map[fq_key] = temp_col
                pre_extracted_temp_cols.append(temp_col)

                df = self._extract_metadata_fields(df, col_name, {struct_key: temp_col})  # ty: ignore[invalid-argument-type]

        for field_spec in self.plan.feature.fields:
            field_key_str = field_spec.key.to_struct_key()
            temp_col_name = f"__concat_{field_key_str}"
            temp_concat_cols[field_key_str] = temp_col_name

            # Build concatenation components
            components: list[nw.Expr] = [
                nw.lit(field_spec.key.to_string()),
                nw.lit(str(field_spec.code_version)),
            ]

            # Add upstream provenance values in deterministic order
            parent_fields = self.plan.get_parent_fields_for_field(field_spec.key)
            for fq_field_key in sorted(parent_fields.keys()):
                components.append(nw.lit(fq_field_key.to_string()))
                col_ref = parent_field_col_map[fq_field_key]
                transformer = self.feature_transformers_by_key.get(fq_field_key.feature)
                expr = nw.col(col_ref)
                if transformer and transformer.is_optional:
                    expr = expr.fill_null("")
                components.append(expr)

            # Concatenate all components
            concat_expr = nw.concat_str(components, separator="|")
            df = df.with_columns(concat_expr.alias(temp_col_name))  # ty: ignore[invalid-argument-type]

        # Hash each concatenation column
        temp_hash_cols: dict[str, str] = {}  # field_key_str -> hash_col_name
        for field_key_str, concat_col in temp_concat_cols.items():
            hash_col_name = f"__hash_{field_key_str}"
            temp_hash_cols[field_key_str] = hash_col_name

            # Hash the concatenated string column into a new column
            df = self.hash_string_column(
                df,  # ty: ignore[invalid-argument-type]
                concat_col,
                hash_col_name,
                hash_algo,
                truncate_length=hash_length,
            )

        # Build provenance_by_field struct
        df = self.build_struct_column(df, METAXY_PROVENANCE_BY_FIELD, temp_hash_cols)  # ty: ignore[invalid-argument-type]

        # Compute sample-level provenance hash
        df = self.hash_struct_version_column(df, hash_algorithm=hash_algo)  # ty: ignore[invalid-assignment]

        # Drop all temporary columns
        temp_columns_to_drop = list(temp_concat_cols.values()) + list(temp_hash_cols.values()) + pre_extracted_temp_cols
        df = df.drop(*temp_columns_to_drop)  # ty: ignore[invalid-argument-type]

        # Drop renamed upstream system columns
        current_columns = df.collect_schema().names()
        columns_to_drop: list[str] = []
        for transformer in self.feature_transformers_by_key.values():
            renamed_prov_col = transformer.renamed_provenance_col
            renamed_prov_by_field_col = transformer.renamed_provenance_by_field_col
            renamed_data_version_by_field_col = transformer.renamed_data_version_by_field_col
            if renamed_prov_col in current_columns:
                columns_to_drop.append(renamed_prov_col)
            if renamed_prov_by_field_col in current_columns:
                columns_to_drop.append(renamed_prov_by_field_col)
            if renamed_data_version_by_field_col in current_columns:
                columns_to_drop.append(renamed_data_version_by_field_col)
            if drop_renamed_data_version_col:
                renamed_data_version_col = transformer.renamed_data_version_col
                if renamed_data_version_col in current_columns:
                    columns_to_drop.append(renamed_data_version_col)

        if columns_to_drop:
            df = df.drop(*columns_to_drop)

        # Add data_version columns (default to provenance values)

        df = df.with_columns(
            nw.col(METAXY_PROVENANCE).alias(METAXY_DATA_VERSION),
            nw.col(METAXY_PROVENANCE_BY_FIELD).alias(METAXY_DATA_VERSION_BY_FIELD),
        )

        return df

    def load_upstream_with_provenance(
        self,
        upstream: dict[FeatureKey, FrameT],
        hash_algo: HashAlgorithm,
        filters: Mapping[FeatureKey, Sequence[nw.Expr]] | None,
    ) -> FrameT:
        """Load and join upstream data, then compute provenance columns.

        This is the main entry point for computing expected provenance from upstream
        features. It prepares upstream DataFrames, joins them, computes provenance
        hashes, and returns a DataFrame with all provenance columns populated.

        Args:
            upstream: Dictionary mapping feature keys to their DataFrames.
            hash_algo: Hash algorithm for provenance computation.
            filters: Optional runtime filters to apply per feature.

        Returns:
            DataFrame with ID columns, data columns, and computed provenance columns.
        """
        df = self.prepare_upstream(upstream, filters=filters, hash_algorithm=hash_algo)

        # Compute provenance columns (shared logic)
        df = self._compute_provenance_internal(
            df,
            hash_algo,
            drop_renamed_data_version_col=True,
        )

        # Drop version columns if present (they come from upstream and shouldn't be in the result)
        version_columns = ["metaxy_feature_version", "metaxy_project_version"]
        current_columns = df.collect_schema().names()  # ty: ignore[invalid-argument-type]
        columns_to_drop = [col for col in version_columns if col in current_columns]

        if columns_to_drop:
            df = df.drop(*columns_to_drop)  # ty: ignore[invalid-argument-type]

        return df

    def compute_provenance_columns(
        self,
        df: FrameT,
        hash_algo: HashAlgorithm,
    ) -> FrameT:
        """Compute provenance columns for a pre-joined DataFrame.

        Use this when you have already joined upstream DataFrames and need to
        compute provenance. The DataFrame must contain renamed upstream metadata
        columns (e.g., metaxy_data_version_by_field__feature_name).

        Args:
            df: Pre-joined DataFrame with renamed upstream metadata columns.
            hash_algo: Hash algorithm for provenance computation.

        Returns:
            DataFrame with provenance columns added.
        """
        return self._compute_provenance_internal(
            df,
            hash_algo,
            drop_renamed_data_version_col=False,
        )

    def hash_struct_version_column(
        self,
        df: FrameT,
        hash_algorithm: HashAlgorithm,
        struct_column: str = METAXY_PROVENANCE_BY_FIELD,
        hash_column: str = METAXY_PROVENANCE,
        field_names: list[str] | None = None,
    ) -> FrameT:
        """Compute sample-level provenance by hashing all field versions together.

        Concatenates all field-level hashes from the struct column and hashes
        the result to produce a single sample-level provenance hash.

        Args:
            df: DataFrame with a struct column containing field-level versions.
            hash_algorithm: Hash algorithm to use.
            struct_column: Name of the struct column to read from.
            hash_column: Name for the output hash column.
            field_names: Field names to include (defaults to all fields in plan).

        Returns:
            DataFrame with the sample-level hash column added.
        """
        if field_names is None:
            field_names = sorted([f.key.to_struct_key() for f in self.plan.feature.fields])

        # Pre-extract fields, concatenate with separator, then hash
        sorted_names = sorted(field_names)
        temp_cols = {fn: f"__hsvf_{fn}" for fn in sorted_names}
        df = self._extract_metadata_fields(df, struct_column, temp_cols)
        sample_concat = nw.concat_str([nw.col(c) for c in temp_cols.values()], separator="|")
        df = df.with_columns(sample_concat.alias("__sample_concat"))  # ty: ignore[invalid-argument-type]

        # Hash the concatenation to produce final provenance hash
        return self.hash_string_column(  # ty: ignore[invalid-return-type]
            df,
            "__sample_concat",
            hash_column,
            hash_algorithm,
            truncate_length=get_hash_truncation_length(),
        ).drop("__sample_concat", *temp_cols.values())

    def resolve_increment_with_provenance(
        self,
        current: FrameT | None,
        upstream: dict[FeatureKey, FrameT],
        hash_algorithm: HashAlgorithm,
        filters: Mapping[FeatureKey, Sequence[nw.Expr]],
        sample: FrameT | None,
        staleness_predicates: tuple[nw.Expr, ...] = (),
    ) -> tuple[FrameT, FrameT | None, FrameT | None, FrameT | None]:
        """Compute expected provenance and compare with current to find changes.

        This is the main entry point for incremental updates. It computes the
        expected provenance from upstream data (or sample for root features),
        then compares with the current stored metadata to identify which samples
        need to be added, updated, or removed.

        Args:
            current: Current metadata from the store, or None for initial load.
            upstream: Dictionary mapping feature keys to their DataFrames.
            hash_algorithm: Hash algorithm for provenance computation.
            filters: Runtime filters to apply per feature.
            sample: For root features, user-provided DataFrame with provenance columns.
            staleness_predicates: Narwhals expressions that identify stale records regardless
                of version. A sample matching any predicate is treated as stale.

        Returns:
            Tuple of (added, changed, removed, input_df) DataFrames. Changed and
            removed may be None if there are no such samples.
        """
        expected, input_df = self._prepare_expected(
            sample,
            upstream,
            hash_algorithm,
            filters,
        )

        # Step 2: Handle case with no current metadata - everything is added
        if current is None:
            return expected, None, None, input_df

        # Step 3: Validate current metadata has required provenance columns
        self._check_required_provenance_columns(
            current,
            "The `current` DataFrame loaded from the metadata store",
        )

        # Step 4: Determine join columns and resolve increment
        join_columns = self.plan.input_id_columns or list(self.plan.feature.id_columns)

        from metaxy.versioning.increment_resolver import IncrementResolver

        resolver: IncrementResolver[FrameT] = IncrementResolver(self.plan, self)
        added, changed, removed = resolver.resolve(expected, current, join_columns, staleness_predicates)

        return added, changed, removed, input_df

    def _prepare_expected(
        self,
        sample: FrameT | None,
        upstream: dict[FeatureKey, FrameT],
        hash_algorithm: HashAlgorithm,
        filters: Mapping[FeatureKey, Sequence[nw.Expr]],
    ) -> tuple[FrameT, FrameT | None]:
        """Prepare the expected dataframe from sample (root features) or upstream."""
        if sample is not None:
            # Root features: sample is user-provided with provenance columns already
            assert len(upstream) == 0, "Root features should have no upstream dependencies"
            expected = sample
            input_df: FrameT | None = None

            # Auto-compute metaxy_provenance if missing but metaxy_provenance_by_field exists
            cols = expected.collect_schema().names()  # ty: ignore[invalid-argument-type]
            if METAXY_PROVENANCE not in cols and METAXY_PROVENANCE_BY_FIELD in cols:
                warnings.warn(
                    f"Auto-computing {METAXY_PROVENANCE} from {METAXY_PROVENANCE_BY_FIELD} because it is missing in samples DataFrame"
                )
                expected = self.hash_struct_version_column(
                    expected,
                    hash_algorithm=hash_algorithm,
                )

            # Validate that root features provide both required provenance columns
            self._check_required_provenance_columns(
                expected,
                "The `sample` DataFrame (must be provided to root features)",
            )
        else:
            # Normal case: compute provenance from upstream
            expected = self.load_upstream_with_provenance(
                upstream,
                hash_algo=hash_algorithm,
                filters=filters,
            )
            # Store input before normalization (for progress calculation)
            input_df = expected

        return expected, input_df

    def _check_required_provenance_columns(self, df: FrameT, message: str):
        cols = df.collect_schema().names()  # ty: ignore[invalid-argument-type]

        if METAXY_PROVENANCE_BY_FIELD not in cols:
            raise ValueError(
                f"{message} is missing required "
                f"'{METAXY_PROVENANCE_BY_FIELD}' column. This column must be a struct containing the provenance of each field on this feature."
            )
        if METAXY_PROVENANCE not in cols:
            raise ValueError(
                f"{message} is missing required "
                f"'{METAXY_PROVENANCE}' column. All metadata in the store must have both provenance columns. "
                f"This column is automatically added by Metaxy when writing metadata."
            )
