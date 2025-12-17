from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import cast

import narwhals as nw
from narwhals.typing import FrameT

from metaxy.config import MetaxyConfig
from metaxy.models.constants import (
    METAXY_FEATURE_SPEC_VERSION,
    METAXY_FEATURE_VERSION,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
    METAXY_SNAPSHOT_VERSION,
)
from metaxy.models.plan import FeaturePlan, FQFieldKey
from metaxy.models.types import FeatureKey, FieldKey
from metaxy.utils.hashing import get_hash_truncation_length
from metaxy.versioning.feature_dep_transformer import FeatureDepTransformer
from metaxy.versioning.renamed_df import RenamedDataFrame
from metaxy.versioning.types import HashAlgorithm


class VersioningEngine(ABC):
    """A class responsible for tracking sample and field level provenance."""

    def __init__(self, plan: FeaturePlan):
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
        transformers = {
            dep.feature: FeatureDepTransformer(dep=dep, plan=self.plan)
            for dep in (self.plan.feature_deps or [])
        }
        # make sure only ID columns are repeated across transformers

        column_counter = Counter()
        all_id_columns = set()
        for transformer in transformers.values():
            renamed_cols = transformer.renamed_columns
            if renamed_cols is not None:
                column_counter.update(renamed_cols)
            all_id_columns.update(transformer.renamed_id_columns)

        repeated_columns = []
        for col, count in column_counter.items():
            if count > 1 and col not in all_id_columns:
                repeated_columns.append(col)

        if repeated_columns:
            raise RuntimeError(
                f"Identified ambiguous columns while resolving upstream column selection for feature {self.key}. Repeated columns: {repeated_columns}. Only ID columns ({all_id_columns}) are allowed to be repeated. Please tweak the `rename` field on the `FeatureDep` objects of {self.key} feature spec."
            )

        return transformers

    @cached_property
    def shared_id_columns(self) -> list[str]:
        """Get the shared ID columns for joining upstream features.

        Returns the ID columns that are common across all upstream features
        after lineage transformations are applied. Each upstream dependency's
        effective ID columns depend on its lineage relationship type:

        - Aggregation lineage uses the aggregation key columns.

        - Identity lineage uses the upstream ID columns.

        - Expansion lineage uses the parent ID columns specified in the relationship.

        The result is the intersection of all effective ID columns across
        upstream dependencies, which determines the join key.

        Example:
            An upstream with `aggregation(on=["group_id"])` contributes only
            `["group_id"]`, while an upstream with identity lineage contributes
            its ID columns. The shared columns are the intersection.
        """
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
        """Join the renamed upstream dataframes on the intersection of renamed id_columns of all feature specs."""
        assert len(upstream) > 0, "No upstream dataframes provided"

        key, renamed_df = next(iter(upstream.items()))

        df = renamed_df.df

        for next_key, renamed_df in upstream.items():
            if key == next_key:
                continue
            # we do not need to provide a _suffix here
            # because the columns are already renamed
            # it's on the user to specify correct renames for colliding columns
            df = cast(
                FrameT, df.join(renamed_df.df, on=self.shared_id_columns, how="inner")
            )

        return df

    def prepare_upstream(
        self,
        upstream: Mapping[FeatureKey, FrameT],
        filters: Mapping[FeatureKey, Sequence[nw.Expr]] | None,
        hash_algorithm: HashAlgorithm | None = None,
    ) -> FrameT:
        """Prepare the upstream dataframes for the given feature.

        This includes, in order:

           - filtering (static filters from FeatureDep.filters + additional runtime filters)

           - renaming

           - selecting

           - lineage transformation (per-dependency, applied independently based on
             [LineageRelationship][metaxy.models.lineage.LineageRelationship])

        based on [FeatureDep][metaxy.models.feature_spec.FeatureDep], and joining
            on the intersection of id_columns of all feature specs.

        Args:
            upstream: Dictionary of upstream dataframes keyed by FeatureKey
            filters: Optional additional runtime filters to apply (combined with FeatureDep.filters)
            hash_algorithm: Hash algorithm for lineage transformations (required for aggregation lineage)
        """
        assert len(upstream) > 0, "No upstream dataframes provided"

        dfs: dict[FeatureKey, RenamedDataFrame[FrameT]] = {
            k: self.feature_transformers_by_key[k].transform(
                df, filters=(filters or {}).get(k)
            )
            for k, df in upstream.items()
        }

        # Apply lineage transformations per-dependency (independently)
        if hash_algorithm is not None:
            from metaxy.versioning.lineage_handler import create_lineage_transformer

            for feature_key, renamed_df in dfs.items():
                dep = self.plan.feature.deps_by_key.get(feature_key)
                if dep is not None:
                    transformer = create_lineage_transformer(dep, self.plan, self)
                    transformed_df = transformer.transform_upstream(
                        renamed_df.df,  # ty: ignore[invalid-argument-type]
                        hash_algorithm,
                    )
                    # Update ID columns based on lineage output
                    dfs[feature_key] = RenamedDataFrame(  # ty: ignore[invalid-assignment]
                        df=transformed_df,
                        id_columns=transformer.output_id_columns,
                    )

        # Drop system columns that aren't needed for provenance calculation
        # Keep only METAXY_PROVENANCE and METAXY_PROVENANCE_BY_FIELD
        # Drop METAXY_FEATURE_VERSION, METAXY_SNAPSHOT_VERSION, and METAXY_FEATURE_SPEC_VERSION
        # to avoid collisions when joining multiple upstream features
        columns_to_drop = [
            METAXY_FEATURE_VERSION,
            METAXY_SNAPSHOT_VERSION,
            METAXY_FEATURE_SPEC_VERSION,
        ]

        for feature_key, renamed_df in dfs.items():
            cols = renamed_df.df.collect_schema().names()  # ty: ignore[invalid-argument-type]
            cols_to_drop = [col for col in columns_to_drop if col in cols]
            if cols_to_drop:
                dfs[feature_key] = RenamedDataFrame(  # ty: ignore[invalid-assignment]
                    df=renamed_df.df.drop(*cols_to_drop),  # ty: ignore[invalid-argument-type]
                    id_columns=renamed_df.id_columns,
                )

        # Validate no column collisions (except ID columns and required system columns)
        if len(dfs) > 1:
            all_columns: dict[str, list[FeatureKey]] = {}
            for feature_key, renamed_df in dfs.items():
                cols = renamed_df.df.collect_schema().names()  # ty: ignore[invalid-argument-type]
                for col in cols:
                    if col not in all_columns:
                        all_columns[col] = []
                    all_columns[col].append(feature_key)

            # System columns that are allowed to collide (needed for provenance calculation)
            from metaxy.models.constants import (
                METAXY_CREATED_AT,
                METAXY_DATA_VERSION,
                METAXY_DATA_VERSION_BY_FIELD,
                METAXY_MATERIALIZATION_ID,
            )

            allowed_system_columns = {
                METAXY_PROVENANCE,
                METAXY_PROVENANCE_BY_FIELD,
                METAXY_DATA_VERSION,
                METAXY_DATA_VERSION_BY_FIELD,
                METAXY_CREATED_AT,
                METAXY_MATERIALIZATION_ID,
            }
            id_cols = set(self.shared_id_columns)
            colliding_columns = [
                col
                for col, features in all_columns.items()
                if len(features) > 1
                and col not in id_cols
                and col not in allowed_system_columns
            ]

            if colliding_columns:
                raise ValueError(
                    f"Found additional shared columns across upstream features for feature {self.plan.feature}: {colliding_columns}. "
                    f"Only ID columns {list(id_cols)} and required system columns {list(allowed_system_columns)} should be shared. "
                    f"Please add explicit renames in your FeatureDep to avoid column collisions."
                )

        return self.join(dfs)  # ty: ignore[invalid-argument-type]

    @abstractmethod
    def hash_string_column(
        self,
        df: FrameT,
        source_column: str,
        target_column: str,
        hash_algo: HashAlgorithm,
    ) -> FrameT:
        """Hash a string column using backend-specific hash function.

        Args:
            df: Narwhals DataFrame
            source_column: Name of string column to hash
            target_column: Name for the new column containing the hash
            hash_algo: Hash algorithm to use
            hash_length: Length to truncate hash to

        Returns:
            Narwhals DataFrame with new hashed column added.
            The source column remains unchanged.
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
            df: Narwhals DataFrame
            struct_name: Name for the new struct column
            field_columns: Mapping from struct field names to column names in df

        Returns:
            Narwhals DataFrame with new struct column added.
            The source columns remain unchanged.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def aggregate_with_string_concat(
        df: FrameT,
        group_by_columns: list[str],
        concat_column: str,
        concat_separator: str,
        exclude_columns: list[str],
    ) -> FrameT:
        """Aggregate DataFrame by grouping and concatenating strings.

        Used for N:1 aggregation lineage where multiple upstream rows
        are aggregated into one downstream row. The concat_column strings
        are concatenated with a separator, and other columns take their
        first value within each group.

        Args:
            df: Narwhals DataFrame to aggregate
            group_by_columns: Columns to group by
            concat_column: Column containing strings to concatenate within groups
            concat_separator: Separator to use when concatenating strings
            exclude_columns: Columns to exclude from aggregation (typically system columns
                that will be recalculated after aggregation)

        Returns:
            Narwhals DataFrame with one row per group, with concat_column containing
            concatenated strings and other columns taking their first value.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def keep_latest_by_group(
        df: FrameT,
        group_columns: list[str],
        timestamp_column: str,
    ) -> FrameT:
        """Keep only the latest row per group based on a timestamp column.

        Args:
            df: Narwhals DataFrame/LazyFrame
            group_columns: Columns to group by (typically ID columns)
            timestamp_column: Column to use for determining "latest" (typically metaxy_created_at)

        Returns:
            Narwhals DataFrame/LazyFrame with only the latest row per group

        Raises:
            ValueError: If timestamp_column doesn't exist in df
        """
        raise NotImplementedError()

    def get_renamed_provenance_by_field_col(self, feature_key: FeatureKey) -> str:
        """Get the renamed provenance_by_field column name for an upstream feature."""
        return self.feature_transformers_by_key[
            feature_key
        ].renamed_provenance_by_field_col

    def get_renamed_data_version_by_field_col(self, feature_key: FeatureKey) -> str:
        """Get the renamed data_version_by_field column name for an upstream feature."""
        return self.feature_transformers_by_key[
            feature_key
        ].renamed_data_version_by_field_col

    def get_field_provenance_exprs(
        self,
    ) -> dict[FieldKey, dict[FQFieldKey, nw.Expr]]:
        """Returns a mapping from field keys to data structures that determine provenances for each field.
        Each value is itself a mapping from fully qualified field keys of upstream features to an expression that selects the corresponding upstream data version.

        Resolves field-level dependencies. Only actual parent fields are considered.

        Note:
            This reads from upstream `metaxy_data_version_by_field` instead of `metaxy_provenance_by_field`,
            enabling users to control version propagation by overriding data_version values.
        """
        res: dict[FieldKey, dict[FQFieldKey, nw.Expr]] = {}
        # THIS LINES HERE
        # ARE THE PINNACLE OF METAXY
        for field_spec in self.plan.feature.fields:
            field_provenance: dict[FQFieldKey, nw.Expr] = {}
            for fq_key, parent_field_spec in self.plan.get_parent_fields_for_field(
                field_spec.key
            ).items():
                # Read from data_version_by_field instead of provenance_by_field
                # This enables user-defined versioning control
                field_provenance[fq_key] = nw.col(
                    self.get_renamed_data_version_by_field_col(fq_key.feature)
                ).struct.field(parent_field_spec.key.to_struct_key())
            res[field_spec.key] = field_provenance
        return res

    def load_upstream_with_provenance(
        self,
        upstream: dict[FeatureKey, FrameT],
        hash_algo: HashAlgorithm,
        filters: Mapping[FeatureKey, Sequence[nw.Expr]] | None,
    ) -> FrameT:
        """Compute the provenance of the given feature.

        Args:
            key: Feature key to compute provenance for
            upstream: Dictionary of upstream dataframes
            hash_algo: Hash algorithm to use
            filters: Optional additional runtime filters to apply to upstream data (combined with FeatureDep.filters)

        Returns:
            DataFrame with metaxy_provenance_by_field and metaxy_provenance columns added

        Note:
            Hash truncation length is read from MetaxyConfig.get().hash_truncation_length
        """
        # Read hash truncation length from global config
        hash_length = MetaxyConfig.get().hash_truncation_length or 64

        # Prepare upstream: filter, rename, select, apply lineage transformations, join
        df = self.prepare_upstream(upstream, filters=filters, hash_algorithm=hash_algo)  # ty: ignore[invalid-argument-type]

        # Build concatenation columns for each field
        temp_concat_cols: dict[str, str] = {}  # field_key_str -> temp_col_name
        field_key_strs: dict[FieldKey, str] = {}  # field_key -> field_key_str

        # Get field provenance expressions
        field_provenance_exprs = self.get_field_provenance_exprs()

        for field_spec in self.plan.feature.fields:
            field_key_str = field_spec.key.to_struct_key()
            field_key_strs[field_spec.key] = field_key_str
            temp_col_name = f"__concat_{field_key_str}"
            temp_concat_cols[field_key_str] = temp_col_name

            # Build concatenation components
            components: list[nw.Expr] = [
                nw.lit(field_spec.key.to_string()),
                nw.lit(str(field_spec.code_version)),
            ]

            # Add upstream provenance values in deterministic order
            parent_field_exprs = field_provenance_exprs.get(field_spec.key, {})
            for fq_field_key in sorted(parent_field_exprs.keys()):
                # Add label
                components.append(nw.lit(fq_field_key.to_string()))
                # Add the expression that selects the upstream provenance
                components.append(parent_field_exprs[fq_field_key])

            # Concatenate all components
            concat_expr = nw.concat_str(components, separator="|")
            df = df.with_columns(concat_expr.alias(temp_col_name))

        # Hash each concatenation column (BACKEND DOES THIS)
        temp_hash_cols: dict[str, str] = {}  # field_key_str -> hash_col_name
        for field_key_str, concat_col in temp_concat_cols.items():
            hash_col_name = f"__hash_{field_key_str}"
            temp_hash_cols[field_key_str] = hash_col_name

            # Hash the concatenated string column into a new column
            df = self.hash_string_column(
                df, concat_col, hash_col_name, hash_algo
            ).with_columns(nw.col(hash_col_name).str.slice(0, hash_length))

        # Build provenance_by_field struct (BACKEND DOES THIS)
        df = self.build_struct_column(df, METAXY_PROVENANCE_BY_FIELD, temp_hash_cols)

        # Compute sample-level provenance hash
        # Step 1: Concatenate all field hashes with separator
        df = self.hash_struct_version_column(df, hash_algorithm=hash_algo)

        # Drop all temporary columns (BASE CLASS CLEANUP)
        # Drop temporary concat columns and hash columns
        temp_columns_to_drop = list(temp_concat_cols.values()) + list(
            temp_hash_cols.values()
        )

        df = df.drop(*temp_columns_to_drop)

        # Drop version columns if present (they come from upstream and shouldn't be in the result)
        version_columns = ["metaxy_feature_version", "metaxy_snapshot_version"]
        current_columns = df.collect_schema().names()
        columns_to_drop = [col for col in version_columns if col in current_columns]

        # Drop renamed upstream provenance and data_version columns (e.g., metaxy_provenance__raw_video)
        # These were needed for provenance calculation but shouldn't be in final result
        for transformer in self.feature_transformers_by_key.values():
            renamed_prov_col = transformer.renamed_provenance_col
            renamed_prov_by_field_col = transformer.renamed_provenance_by_field_col
            renamed_data_version_col = transformer.renamed_data_version_col
            renamed_data_version_by_field_col = (
                transformer.renamed_data_version_by_field_col
            )
            if renamed_prov_col in current_columns:
                columns_to_drop.append(renamed_prov_col)
            if renamed_prov_by_field_col in current_columns:
                columns_to_drop.append(renamed_prov_by_field_col)
            if renamed_data_version_col in current_columns:
                columns_to_drop.append(renamed_data_version_col)
            if renamed_data_version_by_field_col in current_columns:
                columns_to_drop.append(renamed_data_version_by_field_col)

        if columns_to_drop:
            df = df.drop(*columns_to_drop)

        # Add data_version columns (default to provenance values)
        from metaxy.models.constants import (
            METAXY_DATA_VERSION,
            METAXY_DATA_VERSION_BY_FIELD,
        )

        df = df.with_columns(
            nw.col(METAXY_PROVENANCE).alias(METAXY_DATA_VERSION),
            nw.col(METAXY_PROVENANCE_BY_FIELD).alias(METAXY_DATA_VERSION_BY_FIELD),
        )

        return df  # ty: ignore[invalid-return-type]

    def compute_provenance_columns(
        self,
        df: FrameT,
        hash_algo: HashAlgorithm,
    ) -> FrameT:
        """Compute provenance columns for a pre-joined DataFrame.

        This method computes metaxy_provenance_by_field, metaxy_provenance,
        metaxy_data_version_by_field, and metaxy_data_version columns from a
        DataFrame that already contains the renamed upstream metaxy_data_version_by_field
        columns. Use this when you have performed custom joins outside of Metaxy's
        auto-join system.

        The input DataFrame must contain columns named following the pattern
        metaxy_data_version_by_field__<feature_key_parts_joined_by_underscore>
        for each upstream feature dependency. For example, if the upstream feature
        has key ["video", "raw"], the column should be named
        metaxy_data_version_by_field__video_raw.

        Args:
            df: A Narwhals DataFrame or LazyFrame containing pre-joined upstream
                data with renamed metaxy_data_version_by_field columns.
            hash_algo: The hash algorithm to use for computing provenance hashes.

        Returns:
            The input DataFrame with provenance columns added. The renamed upstream
            metaxy_data_version_by_field columns are dropped from the result.
        """
        hash_length = MetaxyConfig.get().hash_truncation_length or 64

        # Build concatenation columns for each field
        temp_concat_cols: dict[str, str] = {}
        field_key_strs: dict[FieldKey, str] = {}

        # Get field provenance expressions
        field_provenance_exprs = self.get_field_provenance_exprs()

        for field_spec in self.plan.feature.fields:
            field_key_str = field_spec.key.to_struct_key()
            field_key_strs[field_spec.key] = field_key_str
            temp_col_name = f"__concat_{field_key_str}"
            temp_concat_cols[field_key_str] = temp_col_name

            # Build concatenation components
            components: list[nw.Expr] = [
                nw.lit(field_spec.key.to_string()),
                nw.lit(str(field_spec.code_version)),
            ]

            # Add upstream provenance values in deterministic order
            parent_field_exprs = field_provenance_exprs.get(field_spec.key, {})
            for fq_field_key in sorted(parent_field_exprs.keys()):
                components.append(nw.lit(fq_field_key.to_string()))
                components.append(parent_field_exprs[fq_field_key])

            concat_expr = nw.concat_str(components, separator="|")
            df = df.with_columns(concat_expr.alias(temp_col_name))  # ty: ignore[invalid-argument-type]

        # Hash each concatenation column
        temp_hash_cols: dict[str, str] = {}
        for field_key_str, concat_col in temp_concat_cols.items():
            hash_col_name = f"__hash_{field_key_str}"
            temp_hash_cols[field_key_str] = hash_col_name

            df = self.hash_string_column(
                df,  # ty: ignore[invalid-argument-type]
                concat_col,
                hash_col_name,
                hash_algo,
            ).with_columns(nw.col(hash_col_name).str.slice(0, hash_length))

        # Build provenance_by_field struct
        df = self.build_struct_column(df, METAXY_PROVENANCE_BY_FIELD, temp_hash_cols)  # ty: ignore[invalid-argument-type]

        # Compute sample-level provenance hash
        df = self.hash_struct_version_column(df, hash_algorithm=hash_algo)  # ty: ignore[invalid-assignment]

        # Drop temporary columns
        temp_columns_to_drop = list(temp_concat_cols.values()) + list(
            temp_hash_cols.values()
        )
        df = df.drop(*temp_columns_to_drop)  # ty: ignore[invalid-argument-type]

        # Drop renamed upstream system columns
        current_columns = df.collect_schema().names()
        columns_to_drop = []
        for transformer in self.feature_transformers_by_key.values():
            renamed_prov_col = transformer.renamed_provenance_col
            renamed_prov_by_field_col = transformer.renamed_provenance_by_field_col
            renamed_data_version_by_field_col = (
                transformer.renamed_data_version_by_field_col
            )
            if renamed_prov_col in current_columns:
                columns_to_drop.append(renamed_prov_col)
            if renamed_prov_by_field_col in current_columns:
                columns_to_drop.append(renamed_prov_by_field_col)
            if renamed_data_version_by_field_col in current_columns:
                columns_to_drop.append(renamed_data_version_by_field_col)

        if columns_to_drop:
            df = df.drop(*columns_to_drop)

        # Add data_version columns (default to provenance values)
        from metaxy.models.constants import (
            METAXY_DATA_VERSION,
            METAXY_DATA_VERSION_BY_FIELD,
        )

        df = df.with_columns(
            nw.col(METAXY_PROVENANCE).alias(METAXY_DATA_VERSION),
            nw.col(METAXY_PROVENANCE_BY_FIELD).alias(METAXY_DATA_VERSION_BY_FIELD),
        )

        return df

    def hash_struct_version_column(
        self,
        df: FrameT,
        hash_algorithm: HashAlgorithm,
        struct_column: str = METAXY_PROVENANCE_BY_FIELD,
        hash_column: str = METAXY_PROVENANCE,
    ) -> FrameT:
        # Compute sample-level provenance from field-level provenance
        # Get all field names from the struct (we need feature spec for this)
        field_names = sorted([f.key.to_struct_key() for f in self.plan.feature.fields])

        # Concatenate all field hashes with separator
        sample_components = [
            nw.col(struct_column).struct.field(field_name) for field_name in field_names
        ]
        sample_concat = nw.concat_str(sample_components, separator="|")
        df = df.with_columns(sample_concat.alias("__sample_concat"))  # ty: ignore[invalid-argument-type]

        # Hash the concatenation to produce final provenance hash
        return (
            self.hash_string_column(  # ty: ignore[invalid-return-type]
                df,
                "__sample_concat",
                hash_column,
                hash_algorithm,
            )
            .with_columns(
                nw.col(hash_column).str.slice(0, get_hash_truncation_length())
            )
            .drop("__sample_concat")
        )

    def resolve_increment_with_provenance(
        self,
        current: FrameT | None,
        upstream: dict[FeatureKey, FrameT],
        hash_algorithm: HashAlgorithm,
        filters: Mapping[FeatureKey, Sequence[nw.Expr]],
        sample: FrameT | None,
    ) -> tuple[FrameT, FrameT | None, FrameT | None, FrameT | None]:
        """Loads upstream data, filters, renames, joins it, calculates expected provenance, and compares it with existing provenance.

        Args:
            current: Current metadata for this feature, if available.
            upstream: A dictionary of upstream data frames.
            hash_algorithm: The hash algorithm to use.
            filters: Additional runtime filters (combined with FeatureDep.filters by FeatureDepTransformer).
            sample: For root features this is used instead of the upstream dataframe.
                Must contain both metaxy_provenance_by_field (struct of field hashes)
                and metaxy_provenance (hash of all field hashes concatenated).
                IMPORTANT: metaxy_provenance must be a HASH, not a raw concatenation.

        Returns:
            tuple[FrameT, FrameT | None, FrameT | None, FrameT | None]
                - added: New samples appearing in upstream. Never None, but may be empty.
                - changed: Samples with changed provenance. May be None for first increment.
                - removed: Samples removed from upstream. May be None for first increment.
                - input: Joined upstream metadata with FeatureDep rules applied. None for root features.

        Note:
            Hash truncation length is read from MetaxyConfig.get().hash_truncation_length
        """
        # Handle root feature case
        if sample is not None:
            # Root features: sample is user-provided with provenance columns already
            assert len(upstream) == 0, (
                "Root features should have no upstream dependencies"
            )
            expected = sample
            input_df: FrameT | None = None  # Root features have no upstream input
            # Auto-compute metaxy_provenance if missing but metaxy_provenance_by_field exists
            cols = expected.collect_schema().names()  # ty: ignore[invalid-argument-type]
            if METAXY_PROVENANCE not in cols and METAXY_PROVENANCE_BY_FIELD in cols:
                warnings.warn(
                    f"Auto-computing {METAXY_PROVENANCE} from {METAXY_PROVENANCE_BY_FIELD} because it is missing in samples DataFrame"
                )
                expected = self.hash_struct_version_column(
                    expected,  # ty: ignore[invalid-argument-type]
                    hash_algorithm=hash_algorithm,
                )

            # Validate that root features provide both required provenance columns
            self._check_required_provenance_columns(
                expected,  # ty: ignore[invalid-argument-type]
                "The `sample` DataFrame (must be provided to root features)",
            )
        else:
            # Normal case: compute provenance from upstream
            expected = self.load_upstream_with_provenance(
                upstream,  # ty: ignore[invalid-argument-type]
                hash_algo=hash_algorithm,
                filters=filters,
            )
            # Store input before normalization (for progress calculation)
            input_df = expected

        # Case 1: No current metadata - everything is added
        if current is None:
            return expected, None, None, input_df
        assert current is not None

        # Case 2 & 3: Compare expected with current metadata
        # Validate that current has metaxy_provenance column
        self._check_required_provenance_columns(
            current,  # ty: ignore[invalid-argument-type]
            "The `current` DataFrame loaded from the metadata store",
        )

        # Join columns = what expected data has after lineage transformations
        # For root features (no deps), fall back to feature's id_columns
        join_columns = self.plan.input_id_columns or list(self.plan.feature.id_columns)

        # Apply per-dependency lineage transformations to current data for comparison
        # Each lineage type may need different handling (e.g., expansion collapses rows)
        from metaxy.versioning.lineage_handler import create_lineage_transformer

        for dep in self.plan.feature_deps or []:
            lineage_transformer = create_lineage_transformer(dep, self.plan, self)
            current = lineage_transformer.transform_current_for_comparison(
                current,  # ty: ignore[invalid-argument-type]
                join_columns,
            )

        current = current.rename(  # ty: ignore[invalid-argument-type]
            {
                METAXY_PROVENANCE: f"__current_{METAXY_PROVENANCE}",
                METAXY_PROVENANCE_BY_FIELD: f"__current_{METAXY_PROVENANCE_BY_FIELD}",
            }
        )

        added = cast(
            FrameT,
            expected.join(  # ty: ignore[invalid-argument-type]
                cast(FrameT, current.select(join_columns)),  # ty: ignore[invalid-argument-type]
                on=join_columns,
                how="anti",
            ),
        )

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
                | (
                    nw.col(METAXY_PROVENANCE)
                    != nw.col(f"__current_{METAXY_PROVENANCE}")
                )
            )
            .drop(f"__current_{METAXY_PROVENANCE}"),
        )

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

        # Return lazy frames with ID and provenance columns (caller decides whether to collect)
        return added, changed, removed, input_df

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
