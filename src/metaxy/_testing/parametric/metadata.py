"""Hypothesis strategies for generating upstream reference metadata for features.

This module provides strategies for property-based testing of features that require
upstream metadata. The generated metadata matches the structure expected by Metaxy's
metadata stores, including all system columns.

Uses Polars' native parametric testing for efficient DataFrame generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from hypothesis import strategies as st
from hypothesis.strategies import composite
from polars.testing.parametric import column, dataframes

from metaxy.config import MetaxyConfig
from metaxy.models.constants import (
    METAXY_CREATED_AT,
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_FEATURE_SPEC_VERSION,
    METAXY_FEATURE_VERSION,
    METAXY_MATERIALIZATION_ID,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
    METAXY_SNAPSHOT_VERSION,
)
from metaxy.models.types import FeatureKey
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.plan import FeaturePlan


from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar, overload

import polars_hash as plh

if TYPE_CHECKING:
    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.plan import FeaturePlan


# Map HashAlgorithm enum to polars-hash functions
_HASH_FUNCTION_MAP: dict[HashAlgorithm, Callable[[pl.Expr], pl.Expr]] = {
    HashAlgorithm.XXHASH64: lambda expr: expr.nchash.xxhash64(),
    HashAlgorithm.XXHASH32: lambda expr: expr.nchash.xxhash32(),
    HashAlgorithm.WYHASH: lambda expr: expr.nchash.wyhash(),
    HashAlgorithm.SHA256: lambda expr: expr.chash.sha2_256(),
    HashAlgorithm.MD5: lambda expr: expr.nchash.md5(),
}


PolarsFrameT = TypeVar("PolarsFrameT", pl.DataFrame, pl.LazyFrame)


@overload
def calculate_provenance_by_field_polars(
    joined_upstream_df: pl.DataFrame,
    feature_spec: FeatureSpec,
    feature_plan: FeaturePlan,
    upstream_column_mapping: dict[str, str],
    hash_algorithm: HashAlgorithm,
    hash_truncation_length: int | None = None,
) -> pl.DataFrame: ...


@overload
def calculate_provenance_by_field_polars(
    joined_upstream_df: pl.LazyFrame,
    feature_spec: FeatureSpec,
    feature_plan: FeaturePlan,
    upstream_column_mapping: dict[str, str],
    hash_algorithm: HashAlgorithm,
    hash_truncation_length: int | None = None,
) -> pl.LazyFrame: ...


def calculate_provenance_by_field_polars(
    joined_upstream_df: pl.DataFrame | pl.LazyFrame,
    feature_spec: FeatureSpec,
    feature_plan: FeaturePlan,
    upstream_column_mapping: dict[str, str],
    hash_algorithm: HashAlgorithm,
    hash_truncation_length: int | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Calculate metaxy_provenance_by_field for a Polars DataFrame.

    This is a standalone function that can be used for testing or direct calculation
    without going through the Narwhals interface.

    Args:
        joined_upstream_df: Polars DataFrame or LazyFrame with upstream data joined
        feature_spec: Feature specification
        feature_plan: Feature plan with field dependencies
        upstream_column_mapping: Maps upstream feature key -> provenance column name
        hash_algorithm: Hash algorithm to use (default: XXHASH64)
        hash_truncation_length: Optional length to truncate hashes to

    Returns:
        Polars frame of the same type as joined_upstream_df with metaxy_provenance_by_field column added

    Example:
        ```python
        from metaxy.data_versioning.calculators.polars import calculate_provenance_by_field_polars
        from metaxy.versioning.types import HashAlgorithm

        result = calculate_provenance_by_field_polars(
            joined_df,
            feature_spec,
            feature_plan,
            upstream_column_mapping={"parent": "metaxy_provenance_by_field"},
            hash_algorithm=HashAlgorithm.SHA256,
            hash_truncation_length=16,
        )
        ```
    """
    if hash_algorithm not in _HASH_FUNCTION_MAP:
        raise ValueError(
            f"Hash algorithm {hash_algorithm} not supported. "
            f"Supported: {list(_HASH_FUNCTION_MAP.keys())}"
        )

    hash_fn = _HASH_FUNCTION_MAP[hash_algorithm]

    # Build hash expressions for each field
    field_exprs = {}

    for field in feature_spec.fields:
        field_key_str = field.key.to_struct_key()

        field_deps = feature_plan.field_dependencies.get(field.key, {})

        # Build hash components
        components = [
            pl.lit(field_key_str),
            pl.lit(str(field.code_version)),
        ]

        # Add upstream provenance values in deterministic order
        for upstream_feature_key in sorted(field_deps.keys()):
            upstream_fields = field_deps[upstream_feature_key]
            upstream_key_str = upstream_feature_key.to_string()

            provenance_col_name = upstream_column_mapping.get(
                upstream_key_str, METAXY_PROVENANCE_BY_FIELD
            )

            for upstream_field in sorted(upstream_fields):
                upstream_field_str = upstream_field.to_struct_key()

                components.append(pl.lit(f"{upstream_key_str}/{upstream_field_str}"))
                components.append(
                    pl.col(provenance_col_name).struct.field(upstream_field_str)
                )

        # Concatenate and hash
        concat_expr = plh.concat_str(*components, separator="|")
        hashed = hash_fn(concat_expr).cast(pl.Utf8)

        # Apply truncation if specified
        if hash_truncation_length is not None:
            hashed = hashed.str.slice(0, hash_truncation_length)

        field_exprs[field_key_str] = hashed

    # Create provenance struct
    provenance_expr = pl.struct(**field_exprs)

    return joined_upstream_df.with_columns(
        provenance_expr.alias(METAXY_PROVENANCE_BY_FIELD)
    )


@composite
def feature_metadata_strategy(
    draw: st.DrawFn,
    feature_spec: FeatureSpec,
    feature_version: str,
    snapshot_version: str,
    num_rows: int | None = None,
    min_rows: int = 1,
    max_rows: int = 100,
    id_columns_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Generate valid metadata DataFrame for a single FeatureSpec.

    Creates a Polars DataFrame with all required Metaxy system columns and ID columns
    as defined in the feature spec. This can be used standalone or as part of
    upstream_metadata_strategy for generating aligned metadata across features.

    Uses Polars' native parametric testing for efficient generation.

    Args:
        draw: Hypothesis draw function (provided by @composite decorator)
        feature_spec: FeatureSpec to generate metadata for
        feature_version: The feature version hash to use (from FeatureGraph)
        snapshot_version: The snapshot version hash to use (from FeatureGraph)
        num_rows: Exact number of rows to generate. If None, will draw from min_rows to max_rows
        min_rows: Minimum number of rows (only used if num_rows is None, default: 1)
        max_rows: Maximum number of rows (only used if num_rows is None, default: 100)
        id_columns_df: Optional DataFrame containing ID column values to use.
            If provided, uses these values and ignores num_rows/min_rows/max_rows.
            Useful for aligning metadata across multiple features in a FeaturePlan.

    Returns:
        Polars DataFrame with ID columns and all Metaxy system columns

    Example:
        ```python
        from hypothesis import given
        from metaxy import FieldSpec, FieldKey
        from metaxy._testing.models import SampleFeatureSpec
        from metaxy._testing.parametric import feature_metadata_strategy

        spec = SampleFeatureSpec(
            key="my_feature",
            fields=[FieldSpec(key=FieldKey(["field1"]))],
        )

        @given(feature_metadata_strategy(spec, min_rows=5, max_rows=20))
        def test_something(metadata_df):
            assert len(metadata_df) >= 5
            assert "sample_uid" in metadata_df.columns
            assert "metaxy_provenance_by_field" in metadata_df.columns
        ```

    Note:
        - The provenance_by_field struct values are generated by Polars
        - System columns use actual Metaxy constant names from models.constants
    """
    # Determine number of rows
    if id_columns_df is not None:
        num_rows_actual = len(id_columns_df)
    elif num_rows is not None:
        num_rows_actual = num_rows
    else:
        num_rows_actual = draw(st.integers(min_value=min_rows, max_value=max_rows))

    # Build list of columns for the DataFrame
    cols = []

    # Add ID columns
    if id_columns_df is not None:
        # Use provided ID column values - we'll add them after generation
        pass
    else:
        # Generate ID columns with Polars
        for id_col in feature_spec.id_columns:
            cols.append(
                column(
                    name=id_col,
                    dtype=pl.Int64,
                    unique=True,  # ID columns should be unique
                    allow_null=False,
                )
            )

    # Add provenance_by_field struct column
    # Use a custom strategy to ensure non-empty strings (hash values shouldn't be empty)
    struct_fields = [
        pl.Field(field_spec.key.to_struct_key(), pl.String)
        for field_spec in feature_spec.fields
    ]

    # Create strategy that generates non-empty hash-like strings
    # Read hash truncation length from global config
    hash_truncation_length = MetaxyConfig.get().hash_truncation_length or 64

    # Generate fixed-length strings matching the truncation length
    hash_string_strategy = st.text(
        alphabet=st.characters(
            whitelist_categories=("Ll", "Nd"),
            whitelist_characters="abcdef0123456789",
        ),
        min_size=hash_truncation_length,
        max_size=hash_truncation_length,
    )

    cols.append(
        column(
            name=METAXY_PROVENANCE_BY_FIELD,
            dtype=pl.Struct(struct_fields),
            strategy=st.builds(
                dict, **{field.name: hash_string_strategy for field in struct_fields}
            ),
            allow_null=False,
        )
    )

    # Generate the DataFrame (without version columns yet)
    df_strategy = dataframes(
        cols=cols,
        min_size=num_rows_actual,
        max_size=num_rows_actual,
    )
    df = draw(df_strategy)

    # Add constant version columns
    df = df.with_columns(  # ty: ignore[unresolved-attribute]
        pl.lit(feature_version).alias(METAXY_FEATURE_VERSION),
        pl.lit(snapshot_version).alias(METAXY_SNAPSHOT_VERSION),
        pl.lit(feature_spec.feature_spec_version).alias(METAXY_FEATURE_SPEC_VERSION),
    )

    # Add METAXY_PROVENANCE column - hash of all field hashes concatenated
    # Get field names from the struct in sorted order for determinism
    field_names = sorted([f.key.to_struct_key() for f in feature_spec.fields])

    # Concatenate all field hashes with separator
    sample_components = [
        pl.col(METAXY_PROVENANCE_BY_FIELD).struct.field(field_name)
        for field_name in field_names
    ]
    sample_concat = plh.concat_str(*sample_components, separator="|")

    # Hash the concatenation using the same algorithm as the test
    hash_fn = _HASH_FUNCTION_MAP.get(HashAlgorithm.XXHASH64)
    if hash_fn is None:
        raise ValueError(f"Hash algorithm {HashAlgorithm.XXHASH64} not supported")

    sample_hash = hash_fn(sample_concat).cast(pl.Utf8)

    # Apply truncation if specified
    if hash_truncation_length is not None:
        sample_hash = sample_hash.str.slice(0, hash_truncation_length)

    df = df.with_columns(sample_hash.alias(METAXY_PROVENANCE))

    # Add data_version columns (default to provenance values)
    df = df.with_columns(
        pl.col(METAXY_PROVENANCE).alias(METAXY_DATA_VERSION),
        pl.col(METAXY_PROVENANCE_BY_FIELD).alias(METAXY_DATA_VERSION_BY_FIELD),
    )

    # Add created_at timestamp column
    from datetime import datetime, timezone

    df = df.with_columns(
        pl.lit(datetime.now(timezone.utc)).alias(METAXY_CREATED_AT),
    )

    # If id_columns_df was provided, replace the generated ID columns with provided ones
    if id_columns_df is not None:
        # Drop the generated ID columns and add the provided ones
        non_id_columns = [
            col for col in df.columns if col not in feature_spec.id_columns
        ]
        df = df.select(non_id_columns)

        # Add the provided ID columns
        for id_col in feature_spec.id_columns:
            if id_col not in id_columns_df.columns:
                raise ValueError(
                    f"ID column '{id_col}' from feature spec not found in id_columns_df. "
                    f"Available columns: {id_columns_df.columns}"
                )
            df = df.with_columns(id_columns_df[id_col])

    return df


@composite
def upstream_metadata_strategy(
    draw: st.DrawFn,
    feature_plan: FeaturePlan,
    feature_versions: dict[str, str],
    snapshot_version: str,
    min_rows: int = 1,
    max_rows: int = 100,
    aggregation_multiplier_min: int = 2,
    aggregation_multiplier_max: int = 5,
) -> dict[str, pl.DataFrame]:
    """Generate upstream reference metadata for a given FeaturePlan.

    Creates a dictionary mapping upstream feature keys to Polars DataFrames that
    contain valid Metaxy metadata. The DataFrames include all system columns
    (metaxy_provenance_by_field, metaxy_feature_version, metaxy_snapshot_version)
    and ID columns as defined in each upstream feature spec.

    Uses Polars' native parametric testing for efficient generation.

    **Lineage-Aware Row Generation:**

    This strategy generates the correct number of rows based on each upstream
    dependency's lineage relationship:

    - **Identity (1:1)**: Same number of rows as the output (min_rows to max_rows)
    - **Aggregation (N:1)**: Multiple rows per output row (aggregation_multiplier_min
      to aggregation_multiplier_max rows per group)
    - **Expansion (1:N)**: Same number of rows as the output (at parent level)

    The generated metadata has the structure expected by metadata stores:
    - ID columns (as defined per feature spec) with generated values
    - metaxy_provenance_by_field: Struct column with field keys mapped to hash strings
    - metaxy_feature_version: Feature version hash string (from FeatureGraph)
    - metaxy_snapshot_version: Snapshot version hash string (from FeatureGraph)

    Args:
        draw: Hypothesis draw function (provided by @composite decorator)
        feature_plan: FeaturePlan containing the feature and its upstream dependencies
        feature_versions: Dict mapping feature key strings to their feature_version hashes
        snapshot_version: The snapshot version hash to use for all features
        min_rows: Minimum number of rows for base (downstream) level (default: 1)
        max_rows: Maximum number of rows for base (downstream) level (default: 100)
        aggregation_multiplier_min: Minimum rows per group for aggregation upstreams (default: 2)
        aggregation_multiplier_max: Maximum rows per group for aggregation upstreams (default: 5)

    Returns:
        Dictionary mapping upstream feature key strings to Polars DataFrames

    Example:
        ```python
        from hypothesis import given
        from metaxy import BaseFeature as FeatureGraph, Feature, FieldSpec, FieldKey
        from metaxy._testing.models import SampleFeatureSpec
        from metaxy._testing.parametric import upstream_metadata_strategy

        graph = FeatureGraph()
        with graph.use():
            class ParentFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key="parent",
                    fields=[FieldSpec(key=FieldKey(["field1"]))],
                ),
            ):
                pass

            class ChildFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key="child",
                    deps=[FeatureDep(feature="parent")],
                    fields=[FieldSpec(key=FieldKey(["result"]))],
                ),
            ):
                pass

            plan = graph.get_feature_plan(FeatureKey(["child"]))

            @given(upstream_metadata_strategy(plan))
            def test_feature_property(upstream_data):
                # upstream_data is a dict with "parent" key mapped to a valid DataFrame
                assert "parent" in upstream_data
                assert "metaxy_provenance_by_field" in upstream_data["parent"].columns
        ```

    Note:
        - The provenance_by_field struct values are generated by Polars
        - Each upstream feature respects its own ID column definition from its spec
        - For joins to work, features with overlapping ID columns will have aligned values
        - System columns use actual Metaxy constant names from models.constants
        - Aggregation upstreams have extra rows per shared ID group
    """
    from metaxy.models.lineage import LineageRelationshipType

    if not feature_plan.deps:
        return {}

    # Generate number of base (downstream) rows
    num_base_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    # Build a mapping from feature_dep.feature to FeatureDep for lineage info
    feature_deps_by_key = {
        dep.feature: dep for dep in (feature_plan.feature_deps or [])
    }

    # Determine the shared ID columns (the columns after lineage transformation)
    # This is the input_id_columns from the plan
    shared_id_columns = feature_plan.input_id_columns or list(
        feature_plan.feature.id_columns
    )

    # Generate a DataFrame with shared ID columns using Polars parametric testing
    shared_id_cols = [
        column(
            name=id_col,
            dtype=pl.Int64,
            unique=True,
            allow_null=False,
        )
        for id_col in sorted(shared_id_columns)  # Sort for deterministic ordering
    ]

    shared_id_columns_df_strategy = dataframes(
        cols=shared_id_cols,
        min_size=num_base_rows,
        max_size=num_base_rows,
    )
    # Note: draw() returns the actual DataFrame from the strategy, but the type
    # system doesn't understand this, so we cast to fix type errors.
    from typing import cast

    shared_id_columns_df = cast(pl.DataFrame, draw(shared_id_columns_df_strategy))

    # Generate metadata for each upstream feature using feature_metadata_strategy
    result: dict[str, pl.DataFrame] = {}

    for upstream_spec in feature_plan.deps:
        # Get the feature version for this upstream feature
        feature_key_str = upstream_spec.key.to_string()
        if feature_key_str not in feature_versions:
            raise ValueError(
                f"Feature version for '{feature_key_str}' not found in feature_versions. "
                f"Available keys: {list(feature_versions.keys())}"
            )
        feature_version = feature_versions[feature_key_str]

        # Get the FeatureDep for this upstream to check lineage
        feature_dep = feature_deps_by_key.get(upstream_spec.key)
        lineage_type = (
            feature_dep.lineage.relationship.type
            if feature_dep
            else LineageRelationshipType.IDENTITY
        )

        # Get the columns shared between upstream and downstream
        # For aggregation: the aggregation columns (a subset of upstream ID columns)
        # For expansion: the parent columns (ExpansionRelationship.on)
        # For identity: the upstream ID columns
        if feature_dep:
            input_cols_for_dep = feature_plan.get_input_id_columns_for_dep(feature_dep)
        else:
            input_cols_for_dep = list(upstream_spec.id_columns)

        # Determine which shared columns exist in this upstream
        cols_to_use = [
            col for col in input_cols_for_dep if col in shared_id_columns_df.columns
        ]

        if lineage_type == LineageRelationshipType.AGGREGATION:
            # For aggregation: need to generate multiple rows per shared ID group
            # The upstream has extra ID columns beyond the shared ones
            aggregation_multiplier = draw(
                st.integers(
                    min_value=aggregation_multiplier_min,
                    max_value=aggregation_multiplier_max,
                )
            )

            # Expand the shared ID columns by repeating each row
            base_shared_df = shared_id_columns_df.select(cols_to_use)
            # Use Polars native repeat to preserve types
            expanded_shared_df = pl.concat(
                [base_shared_df] * aggregation_multiplier, how="vertical"
            ).sort(cols_to_use)

            # Generate extra ID columns that the upstream has but aren't shared
            extra_id_columns = [
                col for col in upstream_spec.id_columns if col not in cols_to_use
            ]

            if extra_id_columns:
                # Generate unique values for extra ID columns
                extra_id_cols = [
                    column(
                        name=id_col,
                        dtype=pl.Int64,
                        unique=True,
                        allow_null=False,
                    )
                    for id_col in extra_id_columns
                ]

                extra_id_df_strategy = dataframes(
                    cols=extra_id_cols,
                    min_size=len(expanded_shared_df),
                    max_size=len(expanded_shared_df),
                )
                extra_id_df = cast(pl.DataFrame, draw(extra_id_df_strategy))

                # Combine shared and extra ID columns
                upstream_id_df = pl.concat(
                    [expanded_shared_df, extra_id_df], how="horizontal"
                )
            else:
                upstream_id_df = expanded_shared_df

        else:
            # For identity and expansion: 1:1 with shared IDs
            # Select only the columns this upstream has
            upstream_id_df = shared_id_columns_df.select(cols_to_use)

            # For upstreams with additional ID columns not in shared, generate them
            extra_id_columns = [
                col for col in upstream_spec.id_columns if col not in cols_to_use
            ]

            if extra_id_columns:
                extra_id_cols = [
                    column(
                        name=id_col,
                        dtype=pl.Int64,
                        unique=True,
                        allow_null=False,
                    )
                    for id_col in extra_id_columns
                ]

                extra_id_df_strategy = dataframes(
                    cols=extra_id_cols,
                    min_size=num_base_rows,
                    max_size=num_base_rows,
                )
                extra_id_df = cast(pl.DataFrame, draw(extra_id_df_strategy))

                upstream_id_df = pl.concat(
                    [upstream_id_df, extra_id_df], how="horizontal"
                )

        df = draw(
            feature_metadata_strategy(
                upstream_spec,
                feature_version=feature_version,
                snapshot_version=snapshot_version,
                id_columns_df=upstream_id_df,
            )
        )

        # Store using feature key string
        result[feature_key_str] = df

    return result


@composite
def downstream_metadata_strategy(
    draw: st.DrawFn,
    feature_plan: FeaturePlan,
    feature_versions: dict[str, str],
    snapshot_version: str,
    hash_algorithm: HashAlgorithm = HashAlgorithm.XXHASH64,
    min_rows: int = 1,
    max_rows: int = 100,
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Generate upstream metadata AND correctly calculated downstream metadata.

    This strategy generates upstream metadata using upstream_metadata_strategy,
    then calculates the "golden" downstream metadata with correctly computed
    metaxy_provenance_by_field values using the Polars calculator.

    This is useful for testing that:
    - Provenance calculations are correct
    - Joins work properly
    - Hash algorithms produce expected results
    - Hash truncation works correctly

    Args:
        draw: Hypothesis draw function (provided by @composite decorator)
        feature_plan: FeaturePlan containing the feature and its upstream dependencies
        feature_versions: Dict mapping feature key strings to their feature_version hashes
            (must include the downstream feature itself)
        snapshot_version: The snapshot version hash to use for all features
        hash_algorithm: Hash algorithm to use for provenance calculation (default: XXHASH64)
        min_rows: Minimum number of rows to generate per upstream feature (default: 1)
        max_rows: Maximum number of rows to generate per upstream feature (default: 100)

    Returns:
        Tuple of (upstream_metadata, downstream_metadata):
        - upstream_metadata: Dict mapping upstream feature keys to DataFrames
        - downstream_metadata: DataFrame with correctly calculated provenance_by_field

    Example:
        ```python
        from hypothesis import given
        from metaxy import BaseFeature as FeatureGraph, FeatureKey
        from metaxy._testing.parametric import downstream_metadata_strategy
        from metaxy.versioning.types import HashAlgorithm

        graph = FeatureGraph()
        # ... define features ...

        plan = graph.get_feature_plan(FeatureKey(["child"]))

        # Get versions from graph
        feature_versions = {
            "parent": graph.get_feature_by_key(FeatureKey(["parent"])).feature_version(),
            "child": graph.get_feature_by_key(FeatureKey(["child"])).feature_version(),
        }
        snapshot_version = graph.snapshot_version()

        @given(downstream_metadata_strategy(
            plan,
            feature_versions=feature_versions,
            snapshot_version=snapshot_version,
            hash_algorithm=HashAlgorithm.SHA256,
        ))
        def test_provenance_calculation(data):
            upstream_data, downstream_df = data
            # Test that downstream_df has correctly calculated provenance
            assert "metaxy_provenance_by_field" in downstream_df.columns
        ```

    Note:
        - The downstream feature's feature_version must be in feature_versions dict
        - Provenance is calculated using the actual Polars calculator
        - Hash algorithm and truncation settings are applied consistently
    """
    # Generate upstream metadata first
    upstream_data = draw(
        upstream_metadata_strategy(
            feature_plan,
            feature_versions={
                k: v
                for k, v in feature_versions.items()
                if k != feature_plan.feature.key.to_string()
            },
            snapshot_version=snapshot_version,
            min_rows=min_rows,
            max_rows=max_rows,
        )
    )

    # For optional dependencies, generate incomplete metadata by dropping some rows.
    # This tests left join behavior where optional upstream samples may be missing.
    optional_dep_keys = {dep.feature.to_string() for dep in feature_plan.optional_deps}
    for feature_key_str, df in upstream_data.items():
        if feature_key_str in optional_dep_keys and len(df) > 1:
            # Drop approximately half of the rows (at least 1, keep at least 1)
            num_to_keep = max(1, len(df) // 2)
            # Use hypothesis to draw which rows to keep for reproducibility
            keep_indices = draw(
                st.lists(
                    st.integers(min_value=0, max_value=len(df) - 1),
                    min_size=num_to_keep,
                    max_size=num_to_keep,
                    unique=True,
                )
            )
            upstream_data[feature_key_str] = df[keep_indices]

    # If there are no upstream features, return empty upstream and just the downstream
    if not upstream_data:
        # Generate standalone downstream metadata
        downstream_feature_key = feature_plan.feature.key.to_string()
        if downstream_feature_key not in feature_versions:
            raise ValueError(
                f"Feature version for downstream feature '{downstream_feature_key}' not found. "
                f"Available keys: {list(feature_versions.keys())}"
            )

        downstream_df = draw(
            feature_metadata_strategy(
                feature_plan.feature,
                feature_version=feature_versions[downstream_feature_key],
                snapshot_version=snapshot_version,
                min_rows=min_rows,
                max_rows=max_rows,
            )
        )
        return ({}, downstream_df)

    # Use the new PolarsVersioningEngine to calculate provenance
    import narwhals as nw

    from metaxy.versioning.polars import PolarsVersioningEngine

    # Create engine (only accepts plan parameter)
    engine = PolarsVersioningEngine(plan=feature_plan)

    # Convert upstream_data keys from strings to FeatureKey objects and wrap in Narwhals
    # Keys are simple strings like "parent", "child" that need to be wrapped in a list
    # DataFrames need to be converted to LazyFrames and wrapped in Narwhals
    upstream_dict = {
        FeatureKey([k]): nw.from_native(v.lazy()) for k, v in upstream_data.items()
    }

    # Load upstream with provenance calculation
    # Note: hash_length is read from MetaxyConfig.get().hash_truncation_length internally
    downstream_df = engine.load_upstream_with_provenance(
        upstream=upstream_dict,
        hash_algo=hash_algorithm,
        filters=None,
    ).collect()

    # Add downstream feature version and snapshot version
    downstream_feature_key = feature_plan.feature.key.to_string()
    if downstream_feature_key not in feature_versions:
        raise ValueError(
            f"Feature version for downstream feature '{downstream_feature_key}' not found. "
            f"Available keys: {list(feature_versions.keys())}"
        )

    # Use Narwhals lit since downstream_df is a Narwhals DataFrame
    from datetime import datetime, timezone

    downstream_df = downstream_df.with_columns(
        nw.lit(feature_versions[downstream_feature_key]).alias(METAXY_FEATURE_VERSION),
        nw.lit(snapshot_version).alias(METAXY_SNAPSHOT_VERSION),
        nw.lit(feature_plan.feature.feature_spec_version).alias(
            METAXY_FEATURE_SPEC_VERSION
        ),
        # Add data_version columns (default to provenance)
        nw.col(METAXY_PROVENANCE).alias(METAXY_DATA_VERSION),
        nw.col(METAXY_PROVENANCE_BY_FIELD).alias(METAXY_DATA_VERSION_BY_FIELD),
        # Add created_at timestamp
        nw.lit(datetime.now(timezone.utc)).alias(METAXY_CREATED_AT),
        # Add materialization_id (nullable)
        nw.lit(None, dtype=nw.String).alias(METAXY_MATERIALIZATION_ID),
    )

    # Convert back to native Polars DataFrame for the return type
    downstream_df_polars = downstream_df.to_native()

    return (upstream_data, downstream_df_polars)
