"""Polars Map dtype conversion utilities for polars-map integration.

Write path: Convert Polars Struct columns to polars_map.Map.
Read path: Reconstruct polars_map.Map columns from List(Struct({key, value})) after reading from stores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

PolarsFrameT = TypeVar("PolarsFrameT", pl.DataFrame, pl.LazyFrame)


def _is_polars_map_dtype(dtype: pl.DataType) -> bool:
    """Check whether a Polars dtype is the ``polars_map.Map`` extension type."""
    return isinstance(dtype, pl.datatypes.classes.BaseExtension) and getattr(dtype, "_name", None) == "polars_map.map"


def convert_structs_to_maps(df: PolarsFrameT, columns: Sequence[str]) -> PolarsFrameT:
    """Convert specified Struct columns to polars_map.Map using expressions.

    Args:
        df: Polars DataFrame or LazyFrame.
        columns: Column names to convert. Only Struct-typed columns in this list
            are converted; non-Struct columns are silently skipped.
    """
    import polars_map  # noqa: F401  # registers .map accessor

    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema  # ty: ignore[invalid-attribute-access]
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


def convert_maps_to_structs(
    df: PolarsFrameT,
    columns: Sequence[str],
    *,
    fallback_field_names: Mapping[str, Sequence[str]] | None = None,
) -> PolarsFrameT:
    """Convert `polars_map.Map` columns to named `Struct` columns.

    This is the inverse of [`convert_structs_to_maps`][] and lets stores that lack a native
    `Map` type persist Metaxy's `Map` columns as `Struct` (or a `Struct`-derived encoding).

    Field names are taken from the keys present in each column. When a column has no keys
    (e.g. an empty frame), `fallback_field_names` supplies the names so the output schema
    stays stable. Columns that are not `Map`-typed are skipped. Native
    `List(Struct({key, value}))` columns are normalized to `Map` first.

    Args:
        df: Polars DataFrame or LazyFrame.
        columns: Column names to convert.
        fallback_field_names: Per-column field names used when a column carries no keys.
    """
    import polars_map  # noqa: F401  # registers the `.map` accessor

    df = convert_maps_to_polars_map(df, columns)
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema  # ty: ignore[invalid-attribute-access]

    struct_exprs: list[pl.Expr] = []
    for col_name in columns:
        if col_name not in schema or not _is_polars_map_dtype(schema[col_name]):
            continue

        keys_expr = pl.col(col_name).map.keys().explode().drop_nulls().unique().sort()  # ty: ignore[unresolved-attribute]
        keys_frame = df.select(keys_expr)  # ty: ignore[invalid-argument-type]
        if isinstance(keys_frame, pl.LazyFrame):
            keys_frame = keys_frame.collect()
        field_names = keys_frame.to_series().to_list()
        if not field_names and fallback_field_names:
            field_names = list(fallback_field_names.get(col_name, []))
        if not field_names:
            continue

        struct_exprs.append(
            pl.struct(
                [pl.col(col_name).map.get(name).alias(name) for name in field_names]  # ty: ignore[unresolved-attribute]
            ).alias(col_name)
        )

    if not struct_exprs:
        return df
    return cast(PolarsFrameT, df.with_columns(struct_exprs))  # ty: ignore[invalid-argument-type]


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
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema  # ty: ignore[invalid-attribute-access]
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
