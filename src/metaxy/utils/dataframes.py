"""DataFrame conversion utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload

import narwhals as nw
import polars as pl
from narwhals.typing import DataFrameT, Frame, FrameT, LazyFrameT

from metaxy._decorators import public

if TYPE_CHECKING:
    import pyarrow as pa

PolarsCompatibleFrame: TypeAlias = Frame | pl.DataFrame | pl.LazyFrame


def find_map_columns(df: Frame) -> list[str]:
    """Return column names that have a Map type in the underlying frame."""
    schema = df.collect_schema() if isinstance(df, nw.LazyFrame) else df.schema
    return [name for name, dtype in schema.items() if isinstance(dtype, nw.Map)]


@public
def collect_to_polars(frame: PolarsCompatibleFrame) -> pl.DataFrame:
    """Helper to convert a frame into an eager Polars DataFrame.

    Args:
        frame: The Narwhals frame to convert.

    Returns:
        The materialized eager Polars DataFrame.
    """
    if isinstance(frame, pl.DataFrame):
        return frame
    if isinstance(frame, pl.LazyFrame):
        return frame.collect()

    if isinstance(frame, (nw.DataFrame, nw.LazyFrame)):
        if frame.implementation == nw.Implementation.POLARS:
            native = frame.to_native()
            return cast(pl.DataFrame, native.collect() if isinstance(native, pl.LazyFrame) else native)

        return frame.collect().to_polars() if isinstance(frame, nw.LazyFrame) else frame.to_polars()

    collected = frame.lazy().collect()
    if isinstance(collected, pl.DataFrame):
        return collected
    return collected.to_polars()


@public
def collect_to_arrow(frame: PolarsCompatibleFrame) -> pa.Table:
    """Convert a frame into a PyArrow Table."""
    nw_frame = nw.from_native(frame) if isinstance(frame, (pl.DataFrame, pl.LazyFrame)) else frame
    collected = nw_frame.collect() if isinstance(nw_frame, nw.LazyFrame) else nw_frame
    return collected.to_arrow()


def lazy_frame_to_polars(frame: nw.LazyFrame[Any]) -> pl.LazyFrame:
    """Helper to convert a Narwhals lazy frame into a Polars lazy frame.

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

    if isinstance(frame, nw.DataFrame):
        return cast(FrameT, nw.from_native(frame.to_polars()))
    if isinstance(frame, nw.LazyFrame):
        return cast(FrameT, nw.from_native(frame.collect().to_polars()).lazy())
    raise ValueError(f"Unsupported frame type: {type(frame)}")
