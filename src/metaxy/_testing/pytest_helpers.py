from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

import narwhals as nw
import polars as pl
import pytest

from metaxy.utils.hashing import get_hash_truncation_length

if TYPE_CHECKING:
    from metaxy.models.feature import BaseFeature
    from metaxy.provenance.types import HashAlgorithm


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
    from metaxy.provenance.polars import PolarsProvenanceTracker
    from metaxy.provenance.types import HashAlgorithm as HashAlgo

    if hash_algorithm is None:
        hash_algorithm = HashAlgo.XXHASH64

    # Get the feature plan from the active graph
    plan = feature.graph.get_feature_plan(feature.spec().key)

    # Create tracker
    tracker = PolarsProvenanceTracker(plan=plan)

    # Convert to Narwhals, add provenance column, convert back
    df_nw = nw.from_native(df.lazy())
    df_nw = tracker.add_provenance_column(df_nw, hash_algorithm=hash_algorithm)
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
                # If exception of given type happens
                # just swallow it and raise pytest.Skip with given reason
                pytest.skip(f"skipped {exception.__name__}: {reason}")

        return wrapper

    return decorator_func
