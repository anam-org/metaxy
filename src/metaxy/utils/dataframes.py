"""DataFrame conversion utilities."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload

import narwhals as nw
import polars as pl
from narwhals.typing import DataFrameT, Frame, FrameT, LazyFrameT

from metaxy._decorators import public

if TYPE_CHECKING:
    import pyarrow as pa

PolarsCompatibleFrame: TypeAlias = Frame | pl.DataFrame | pl.LazyFrame


def _is_polars_map_dtype(dtype: pl.DataType) -> bool:
    """Check if a Polars dtype is a polars_map.Map extension type without importing polars_map."""
    return isinstance(dtype, pl.datatypes.classes.BaseExtension) and getattr(dtype, "_name", None) == "polars_map.map"


def find_map_columns(df: Frame) -> list[str]:
    """Return column names that have a Map type in the underlying frame.

    Returns empty list when enable_map_datatype is not set.
    Inspects the native backend (Polars, Arrow, Ibis) for Map-typed columns.
    """
    from metaxy.config import MetaxyConfig

    if not MetaxyConfig.get().enable_map_datatype:
        return []

    if df.implementation == nw.Implementation.POLARS:
        native = df.to_native()
        schema = native.collect_schema() if isinstance(native, pl.LazyFrame) else native.schema
        return [name for name, dtype in schema.items() if _is_polars_map_dtype(dtype)]

    if df.implementation == nw.Implementation.PYARROW:
        import pyarrow as pa

        table = cast(pa.Table, df.to_native())
        return [field.name for field in table.schema if pa.types.is_map(field.type)]

    if df.implementation == nw.Implementation.IBIS:
        import ibis
        import ibis.expr.datatypes as dt

        table = cast(ibis.Table, df.to_native())
        return [name for name, dtype in table.schema().items() if isinstance(dtype, dt.Map)]

    warnings.warn(
        f"cannot identify Map columns for unsupported narwhals implementation {df.implementation}",
        stacklevel=2,
    )
    return []


def has_polars_map_columns(df: pl.DataFrame | pl.LazyFrame) -> bool:
    """Check if a Polars DataFrame/LazyFrame has any polars_map.Map columns."""
    return bool(find_map_columns(cast(Frame, nw.from_native(df))))


@public
def collect_to_polars(frame: PolarsCompatibleFrame) -> pl.DataFrame:
    """Helper to convert a frame into an eager Polars DataFrame.

    Preserves `Map` columns as `polars_map.Map` when `MetaxyConfig.enable_map_datatype` is set.

    Args:
        frame: The Narwhals frame to convert.

    Returns:
        The materialized eager Polars DataFrame.
    """
    if isinstance(frame, pl.DataFrame):
        return frame
    if isinstance(frame, pl.LazyFrame):
        return cast(pl.DataFrame, frame.collect())

    if isinstance(frame, (nw.DataFrame, nw.LazyFrame)):
        if frame.implementation == nw.Implementation.POLARS:
            native = frame.to_native()
            return cast(pl.DataFrame, native.collect() if isinstance(native, pl.LazyFrame) else native)

        map_cols = find_map_columns(frame)
        result = frame.collect().to_polars() if isinstance(frame, nw.LazyFrame) else frame.to_polars()
        if map_cols:
            from metaxy.versioning._arrow_map import convert_maps_to_polars_map

            result = convert_maps_to_polars_map(result, columns=map_cols)
        return result

    collected = frame.lazy().collect()
    if isinstance(collected, pl.DataFrame):
        return collected
    return collected.to_polars()


@public
def collect_to_arrow(frame: PolarsCompatibleFrame) -> pa.Table:
    """Convert a frame into a PyArrow Table.

    If the dataframe is a Polars dataframe, it converts `polars_map.Map`
        extension columns to native Arrow `MapArray` when `MetaxyConfig.enable_map_datatype` is set.
    """
    nw_frame = nw.from_native(frame) if isinstance(frame, (pl.DataFrame, pl.LazyFrame)) else frame
    collected = nw_frame.collect() if isinstance(nw_frame, nw.LazyFrame) else nw_frame
    table: pa.Table = collected.to_arrow()

    if collected.implementation == nw.Implementation.POLARS:
        from metaxy.versioning._arrow_map import convert_extension_maps_to_native, has_extension_map_columns

        if has_extension_map_columns(table):
            table = convert_extension_maps_to_native(table)

    return table


def lazy_frame_to_polars(frame: nw.LazyFrame[Any]) -> pl.LazyFrame:
    """Helper to convert a Narwhals lazy frame into a Polars lazy frame.

    Preserves `polars_map.Map` columns when `MetaxyConfig.enable_map_datatype` is set.
    If the Narwhals LazyFrame is already backed by Polars, this is a no-op.
    """
    if frame.implementation == nw.Implementation.POLARS:
        return frame.to_native()
    return collect_to_polars(frame).lazy()


@overload
def switch_implementation_to_polars(frame: DataFrameT) -> DataFrameT: ...


@overload
def switch_implementation_to_polars(frame: LazyFrameT) -> LazyFrameT: ...


def switch_implementation_to_polars(frame: FrameT) -> FrameT:
    if frame.implementation == nw.Implementation.POLARS:
        return frame

    # Detect map columns before conversion (they lose their type in Polars)
    from metaxy.config import MetaxyConfig

    map_columns: list[str] = []
    if MetaxyConfig.get().enable_map_datatype:
        map_columns = find_map_columns(frame)

    if isinstance(frame, nw.DataFrame):
        result = nw.from_native(frame.to_polars())
    elif isinstance(frame, nw.LazyFrame):
        result = nw.from_native(frame.collect().to_polars()).lazy()
    else:
        raise ValueError(f"Unsupported frame type: {type(frame)}")

    # Reconstruct polars_map.Map columns that were lost during conversion
    if map_columns:
        from metaxy.versioning._arrow_map import convert_maps_to_polars_map

        native = result.to_native()
        native = convert_maps_to_polars_map(native, columns=map_columns)
        result = nw.from_native(native)

    return result
