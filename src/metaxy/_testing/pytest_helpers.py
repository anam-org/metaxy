from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, TypeVar

import narwhals as nw
import polars as pl
from narwhals.typing import IntoFrameT

from metaxy.models.constants import (
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.utils.hashing import get_hash_truncation_length

if TYPE_CHECKING:
    from metaxy.models.feature import BaseFeature
    from metaxy.versioning.types import HashAlgorithm

FrameT = TypeVar("FrameT", bound=nw.DataFrame | nw.LazyFrame)


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
    df_nw = nw.from_native(df)
    columns = df_nw.collect_schema().names()  # ty: ignore[possibly-missing-attribute]

    columns_to_add: list[nw.Expr] = []

    # If metaxy_provenance_by_field doesn't exist, we can't derive the others meaningfully
    # In that case, add placeholders
    if METAXY_PROVENANCE_BY_FIELD not in columns:
        if METAXY_PROVENANCE not in columns:
            columns_to_add.append(nw.lit("test_provenance").alias(METAXY_PROVENANCE))
        if METAXY_DATA_VERSION not in columns:
            columns_to_add.append(
                nw.lit("test_data_version").alias(METAXY_DATA_VERSION)
            )
        # Can't add metaxy_data_version_by_field without knowing the struct schema
    else:
        # metaxy_provenance_by_field exists, derive others from it
        if METAXY_PROVENANCE not in columns:
            columns_to_add.append(nw.lit("test_provenance").alias(METAXY_PROVENANCE))

        if METAXY_DATA_VERSION_BY_FIELD not in columns:
            # Copy provenance_by_field to data_version_by_field
            columns_to_add.append(
                nw.col(METAXY_PROVENANCE_BY_FIELD).alias(METAXY_DATA_VERSION_BY_FIELD)
            )

        if METAXY_DATA_VERSION not in columns:
            # If provenance exists, copy it; otherwise use placeholder
            if METAXY_PROVENANCE in columns:
                columns_to_add.append(
                    nw.col(METAXY_PROVENANCE).alias(METAXY_DATA_VERSION)
                )
            else:
                columns_to_add.append(
                    nw.lit("test_data_version").alias(METAXY_DATA_VERSION)
                )

    if columns_to_add:
        df_nw = df_nw.with_columns(columns_to_add)  # ty: ignore[possibly-missing-attribute]

    return df_nw.to_native()  # ty: ignore[possibly-missing-attribute]


def add_metaxy_provenance_column(
    df: pl.DataFrame,
    feature: type[BaseFeature],
    hash_algorithm: HashAlgorithm | None = None,
) -> pl.DataFrame:
    """Add metaxy_provenance column to a DataFrame based on metaxy_provenance_by_field.


    Args:
        df: Polars DataFrame with metaxy_provenance_by_field column
        feature: Feature class to get the feature plan from
        hash_algorithm: Hash algorithm to use. If None, uses XXHASH64.

    Returns:
        Polars DataFrame with metaxy_provenance column added
    """
    from metaxy.versioning.polars import PolarsVersioningEngine
    from metaxy.versioning.types import HashAlgorithm as HashAlgo

    if hash_algorithm is None:
        hash_algorithm = HashAlgo.XXHASH64

    # Get the feature plan from the active graph
    plan = feature.graph.get_feature_plan(feature.spec().key)

    # Create engine
    engine = PolarsVersioningEngine(plan=plan)

    # Convert to Narwhals, add provenance column, convert back
    df_nw = nw.from_native(df.lazy())
    df_nw = engine.hash_struct_version_column(df_nw, hash_algorithm=hash_algorithm)
    result_df = df_nw.collect().to_native()

    # Apply hash truncation if specified
    result_df = result_df.with_columns(
        pl.col("metaxy_provenance").str.slice(0, get_hash_truncation_length())
    )

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
