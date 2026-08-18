from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, TypeVar

import narwhals as nw
import polars as pl
from metaxy._hashing import get_hash_truncation_length
from metaxy.models.constants import (
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.utils._arrow_map import convert_maps_to_structs, convert_structs_to_maps
from narwhals.typing import IntoFrameT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from metaxy import CoercibleToFeatureKey
    from metaxy.models.feature import BaseFeature
    from metaxy.models.feature_definition import FeatureDefinition
    from metaxy.versioning.types import HashAlgorithm

FrameT = TypeVar("FrameT", bound=nw.DataFrame | nw.LazyFrame)
PolarsFrameT = TypeVar("PolarsFrameT", pl.DataFrame, pl.LazyFrame)

# The metaxy system columns that the versioning engine and stores emit as the
# ``Map(String, String)`` dtype (always-on Map behavior).
BY_FIELD_MAP_COLUMNS: tuple[str, ...] = (METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD)


def normalize_by_field_maps(
    df: PolarsFrameT,
    columns: Sequence[str] = BY_FIELD_MAP_COLUMNS,
) -> PolarsFrameT:
    """Canonicalize metaxy ``*_by_field`` columns to the ``Map`` dtype.

    Stores and the versioning engine emit these columns as ``Map``, while test data
    may build them as ``Struct``. This normalizes both representations to ``Map`` so
    both sides of a comparison (or a frame about to be written) share one representation.
    Only the requested columns are touched, and only when they need converting.
    """
    return convert_structs_to_maps(df, columns)


def assert_metadata_frame_equal(left: pl.DataFrame, right: pl.DataFrame, **kwargs: object) -> None:
    """`polars.testing.assert_frame_equal` after canonicalizing metaxy `*_by_field` columns.

    Metadata frames carry the `*_by_field` columns in one of several equivalent representations
    (`Struct` or `Map`) depending on how they were built
    or materialized. Both sides are canonicalized to `Map` so the comparison checks
    values rather than incidental dtype. Extra keyword arguments are forwarded to
    `assert_frame_equal` (e.g. `check_row_order`, `check_column_order`).
    """
    from polars.testing import assert_frame_equal

    assert_frame_equal(normalize_by_field_maps(left), normalize_by_field_maps(right), **kwargs)  # ty: ignore[invalid-argument-type]


def by_field_maps_to_structs(
    df: pl.DataFrame,
    columns: Sequence[str] = BY_FIELD_MAP_COLUMNS,
) -> pl.DataFrame:
    """Convert metaxy ``*_by_field`` Map columns back to ``Struct`` for element access.

    Map scalars do not support ``value["field"]`` indexing, so tests that inspect
    individual field hashes convert the Map columns to Struct first. Accepts Struct
    or ``Map`` inputs.
    """
    return convert_maps_to_structs(normalize_by_field_maps(df, columns), columns)


def add_metaxy_system_columns(df: IntoFrameT) -> IntoFrameT:
    """Add missing metaxy system columns to a DataFrame for testing.

    This function adds the metaxy_provenance, metaxy_data_version, and
    metaxy_data_version_by_field columns based on existing metaxy_provenance_by_field.

    If metaxy_provenance_by_field is missing, adds placeholder values.

    Works with Polars DataFrames/LazyFrames and Narwhals DataFrames/LazyFrames.

    Args:
        df: DataFrame to add columns to (Polars or Narwhals)

    Returns:
        DataFrame with all required metaxy system columns (same type as input)
    """
    # Convert to narwhals if needed (works with both eager and lazy frames)
    df_nw = nw.from_native(df)  # ty: ignore[no-matching-overload]
    columns = df_nw.collect_schema().names()

    columns_to_add: list[nw.Expr] = []

    # If metaxy_provenance_by_field doesn't exist, we can't derive the others meaningfully
    # In that case, add placeholders
    if METAXY_PROVENANCE_BY_FIELD not in columns:
        if METAXY_PROVENANCE not in columns:
            columns_to_add.append(nw.lit("test_provenance").alias(METAXY_PROVENANCE))
        if METAXY_DATA_VERSION not in columns:
            columns_to_add.append(nw.lit("test_data_version").alias(METAXY_DATA_VERSION))
        # Can't add metaxy_data_version_by_field without knowing the struct schema
    else:
        # metaxy_provenance_by_field exists, derive others from it
        if METAXY_PROVENANCE not in columns:
            columns_to_add.append(nw.lit("test_provenance").alias(METAXY_PROVENANCE))

        if METAXY_DATA_VERSION_BY_FIELD not in columns:
            # Copy provenance_by_field to data_version_by_field
            columns_to_add.append(nw.col(METAXY_PROVENANCE_BY_FIELD).alias(METAXY_DATA_VERSION_BY_FIELD))

        if METAXY_DATA_VERSION not in columns:
            # If provenance exists, copy it; otherwise use placeholder
            if METAXY_PROVENANCE in columns:
                columns_to_add.append(nw.col(METAXY_PROVENANCE).alias(METAXY_DATA_VERSION))
            else:
                columns_to_add.append(nw.lit("test_data_version").alias(METAXY_DATA_VERSION))

    if columns_to_add:
        df_nw = df_nw.with_columns(columns_to_add)

    return df_nw.to_native()


def add_metaxy_provenance_column(
    df: pl.DataFrame,
    feature: type[BaseFeature] | FeatureDefinition | CoercibleToFeatureKey,
    hash_algorithm: HashAlgorithm | None = None,
) -> pl.DataFrame:
    """Add metaxy_provenance column to a DataFrame based on metaxy_provenance_by_field.


    Args:
        df: Polars DataFrame with metaxy_provenance_by_field column
        feature: Feature class, FeatureDefinition, or value coercible to FeatureKey
        hash_algorithm: Hash algorithm to use. If None, uses XXHASH64.

    Returns:
        Polars DataFrame with metaxy_provenance column added
    """
    from metaxy import coerce_to_feature_key
    from metaxy.ext.polars.versioning import PolarsVersioningEngine
    from metaxy.models.feature import current_graph
    from metaxy.versioning.types import HashAlgorithm as HashAlgo

    if hash_algorithm is None:
        hash_algorithm = HashAlgo.XXHASH64

    # Coerce to feature key and get plan from active graph
    feature_key = coerce_to_feature_key(feature)
    graph = current_graph()
    plan = graph.get_feature_plan(feature_key)

    # Create engine
    engine = PolarsVersioningEngine(plan=plan)

    # Convert to Narwhals, add provenance column, convert back
    df_nw = nw.from_native(df.lazy())
    df_nw = engine.hash_struct_version_column(df_nw, hash_algorithm=hash_algorithm)
    result_df = df_nw.collect().to_native()

    # Apply hash truncation if specified
    result_df = result_df.with_columns(pl.col("metaxy_provenance").str.slice(0, get_hash_truncation_length()))

    return result_df


def skip_exception(exception: type[Exception], reason: str):
    # Func below is the real decorator and will receive the test function as param
    def decorator_func(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                # Try to run the test
                return f(*args, **kwargs)
            except exception:
                import pytest

                # If exception of given type happens
                # just swallow it and raise pytest.Skip with given reason
                pytest.skip(f"skipped {exception.__name__}: {reason}")

        return wrapper

    return decorator_func
