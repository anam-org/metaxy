"""Conversion utilities between Polars `Struct` and native Polars `Map` columns."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

PolarsFrameT = TypeVar("PolarsFrameT", pl.DataFrame, pl.LazyFrame)


def convert_structs_to_maps(df: PolarsFrameT, columns: Sequence[str]) -> PolarsFrameT:
    """Convert specified Struct columns to `Map(String, value)` columns.

    Args:
        df: Polars DataFrame or LazyFrame.
        columns: Column names to convert. Only Struct-typed columns in this list
            are converted; non-Struct columns are silently skipped.
    """
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
    map_exprs: list[pl.Expr] = []

    for col_name in columns:
        if col_name not in schema:
            continue
        dtype = schema[col_name]
        if not isinstance(dtype, pl.Struct):
            continue

        entries = pl.concat_list(
            [
                pl.struct(
                    pl.lit(field.name).alias("key"),
                    pl.col(col_name).struct.field(field.name).alias("value"),
                )
                for field in dtype.fields
            ]
        ).alias(col_name)
        # The value dtype is the supertype of all Struct field dtypes, as resolved by `concat_list`.
        entries_dtype = cast(pl.List, pl.LazyFrame(schema={col_name: dtype}).select(entries).collect_schema()[col_name])
        value_dtype = cast(pl.Struct, entries_dtype.inner).to_schema()["value"]
        map_exprs.append(entries.cast(pl.Map(pl.String, value_dtype)))

    if not map_exprs:
        return df
    return cast(PolarsFrameT, df.with_columns(map_exprs))  # ty: ignore[invalid-argument-type]


def convert_maps_to_structs(
    df: PolarsFrameT,
    columns: Sequence[str],
    *,
    fallback_field_names: Mapping[str, Sequence[str]] | None = None,
) -> PolarsFrameT:
    """Convert `Map` columns to named `Struct` columns.

    This is the inverse of [`convert_structs_to_maps`][] and lets stores that lack a native
    `Map` type persist Metaxy's `Map` columns as `Struct` (or a `Struct`-derived encoding).

    Field names are taken from the keys present in each column. When a column has no keys
    (e.g. an empty frame), `fallback_field_names` supplies the names so the output schema
    stays stable. Columns that are not `Map`-typed are skipped.

    Args:
        df: Polars DataFrame or LazyFrame.
        columns: Column names to convert.
        fallback_field_names: Per-column field names used when a column carries no keys.
    """
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema

    struct_exprs: list[pl.Expr] = []
    for col_name in columns:
        if col_name not in schema or not isinstance(schema[col_name], pl.Map):
            continue

        keys_expr = pl.col(col_name).map.keys().explode().drop_nulls().unique().sort()
        keys_frame = df.select(keys_expr)  # ty: ignore[invalid-argument-type]
        if isinstance(keys_frame, pl.LazyFrame):
            keys_frame = keys_frame.collect()
        field_names = keys_frame.to_series().to_list()
        if not field_names and fallback_field_names:
            field_names = list(fallback_field_names.get(col_name, []))
        if not field_names:
            continue

        struct_exprs.append(
            pl.struct([pl.col(col_name).map.get(name).alias(name) for name in field_names]).alias(col_name)
        )

    if not struct_exprs:
        return df
    return cast(PolarsFrameT, df.with_columns(struct_exprs))  # ty: ignore[invalid-argument-type]
