"""Arrow Map conversion utilities for polars-map integration.

Write path: Convert Polars Struct columns to native Arrow MapArray before writing to stores.
Read path: Reconstruct polars_map.Map columns from Arrow MapArray after reading from stores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

import polars as pl
import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Sequence

PolarsFrameT = TypeVar("PolarsFrameT", pl.DataFrame, pl.LazyFrame)


# ── Write path: Struct → polars_map.Map → native Arrow MapArray ──────


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


def convert_extension_maps_to_native(table: pa.Table) -> pa.Table:
    """Replace polars-map extension-typed columns with native Arrow MapArray columns.

    When a Polars DataFrame with polars_map.Map columns is converted to Arrow,
    those columns carry ARROW:extension:name metadata with List(Struct({key,value}))
    storage. This converts them to native Arrow MapArray for store compatibility.
    """
    for i, field in enumerate(table.schema):
        if not _is_extension_map_field(field):
            continue
        col = table.column(i)
        native_chunks = [_extension_chunk_to_native_map(chunk) for chunk in col.chunks]
        native_col = pa.chunked_array(native_chunks)
        table = table.set_column(i, pa.field(field.name, native_col.type), native_col)
    return table


def _is_extension_map_field(field: pa.Field) -> bool:
    """Check if an Arrow field has polars-map extension metadata."""
    metadata = field.metadata
    if metadata is None:
        return False
    return metadata.get(b"ARROW:extension:name") == b"polars_map.map"


def _extension_chunk_to_native_map(arr: pa.Array) -> pa.MapArray:
    """Convert a polars-map extension chunk (LargeList of Struct({key,value})) to native MapArray."""
    # The array is a (Large)ListArray with Struct({key, value}) values
    # Downcast large_string → string for MapArray compatibility, preserve other types
    keys = arr.values.field("key")
    if pa.types.is_large_string(keys.type):
        keys = keys.cast(pa.string())
    values = arr.values.field("value")
    if pa.types.is_large_string(values.type):
        values = values.cast(pa.string())
    offsets = arr.offsets.cast(pa.int32())
    return pa.MapArray.from_arrays(offsets, keys, values)


def has_extension_map_columns(table: pa.Table) -> bool:
    """Check if an Arrow table has any polars-map extension-typed columns."""
    return any(_is_extension_map_field(field) for field in table.schema)


def strip_extension_map_metadata(table: pa.Table) -> pa.Table:
    """Strip polars-map extension metadata, keeping data as List(Struct({key, value})).

    For backends that don't support native Arrow MapArray (e.g. LanceDB/Lance),
    this preserves the data in a compatible format while removing extension metadata
    that would confuse the storage layer.
    """
    for i, field in enumerate(table.schema):
        if not _is_extension_map_field(field):
            continue
        col = table.column(i)
        storage_type = col.type
        if hasattr(storage_type, "storage_type"):
            storage_type = storage_type.storage_type
        clean_field = pa.field(field.name, storage_type)
        storage_chunks = [chunk.storage if hasattr(chunk, "storage") else chunk for chunk in col.chunks]
        table = table.set_column(i, clean_field, pa.chunked_array(storage_chunks))
    return table


# ── Read path: Arrow MapArray → polars_map.Map ──────────────────────────


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
