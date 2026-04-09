"""Polars Map dtype conversion utilities for polars-map integration.

Write path: Convert Polars Struct columns to polars_map.Map.
Read path: Reconstruct polars_map.Map columns from List(Struct({key, value})) after reading from stores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

PolarsFrameT = TypeVar("PolarsFrameT", pl.DataFrame, pl.LazyFrame)


def convert_structs_to_maps(df: PolarsFrameT, columns: Sequence[str]) -> PolarsFrameT:
    """Convert specified Struct columns to polars_map.Map using expressions.

    Args:
        df: Polars DataFrame or LazyFrame.
        columns: Column names to convert. Only Struct-typed columns in this list
            are converted; non-Struct columns are silently skipped.
    """
    import polars_map  # noqa: F401  # registers .map accessor

    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
    map_exprs: list[pl.Expr] = []

    for col_name in columns:
        if col_name not in schema:
            continue
        dtype = schema[col_name]
        if not isinstance(dtype, pl.Struct):
            continue

        field_names = [f.name for f in dtype.fields]
        kv_pairs = [
            pl.struct(
                pl.lit(name).alias("key"),
                pl.col(col_name).struct.field(name).alias("value"),
            )
            for name in field_names
        ]
        map_exprs.append(pl.concat_list(kv_pairs).map.from_entries().alias(col_name))  # ty: ignore[unresolved-attribute]

    if not map_exprs:
        return df
    return cast(PolarsFrameT, df.with_columns(map_exprs))  # ty: ignore[invalid-argument-type]


def convert_maps_to_polars_map(
    df: PolarsFrameT,
    columns: Sequence[str],
) -> PolarsFrameT:
    """Reconstruct polars_map.Map columns from List(Struct({key, value})) columns.

    When Polars reads native Arrow MapArray columns, they appear as
    List(Struct({key, value})). This function reconstructs them as
    polars_map.Map extension type columns.

    Args:
        df: Polars DataFrame or LazyFrame.
        columns: Column names to convert. Only columns with List(Struct({key, value}))
            dtype are converted; others are silently skipped.
    """
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
    target_columns: list[str] = []

    for col_name, dtype in schema.items():
        if col_name not in columns:
            continue
        if not _is_list_of_kv_struct(dtype):
            continue
        target_columns.append(col_name)

    if not target_columns:
        return df

    import polars_map  # noqa: F401  # registers .map accessor

    map_exprs = [
        pl.col(col_name).map.from_entries().alias(col_name)  # ty: ignore[unresolved-attribute]
        for col_name in target_columns
    ]
    return cast(PolarsFrameT, df.with_columns(map_exprs))  # ty: ignore[invalid-argument-type]


def _is_list_of_kv_struct(dtype: pl.DataType) -> bool:
    """Check if a Polars dtype is List(Struct({key: String, value: String}))."""
    if not isinstance(dtype, pl.List):
        return False
    inner = dtype.inner
    if not isinstance(inner, pl.Struct):
        return False
    field_names = {f.name for f in inner.fields}
    return field_names == {"key", "value"}
