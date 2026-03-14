from typing import Any, TypeAlias, cast, overload

import narwhals as nw
import polars as pl
from narwhals.typing import DataFrameT, Frame, FrameT, LazyFrameT

PolarsCompatibleFrame: TypeAlias = Frame | pl.DataFrame | pl.LazyFrame


def collect_to_polars(frame: PolarsCompatibleFrame) -> pl.DataFrame:
    """Helper to convert a Narwhals frame into an eager Polars DataFrame.

    Avoids unnecessary re-materialization when the frame is already a Polars-backed
    eager DataFrame.
    """
    if isinstance(frame, nw.DataFrame):
        if frame.implementation == nw.Implementation.POLARS:
            return cast(pl.DataFrame, frame.to_native())
        return frame.to_polars()

    if isinstance(frame, nw.LazyFrame):
        return cast(pl.DataFrame, lazy_frame_to_polars(frame).collect())

    if isinstance(frame, pl.DataFrame):
        return frame

    if isinstance(frame, pl.LazyFrame):
        return cast(pl.DataFrame, frame.collect())

    collected = frame.lazy().collect()
    if isinstance(collected, pl.DataFrame):
        return collected
    return collected.to_polars()


def lazy_frame_to_polars(frame: nw.LazyFrame[Any]) -> pl.LazyFrame:
    """Helper to convert a Narwhals lazy frame into a Polars lazy frame.

    If the Narwhals LazyFrame is already backed by Polars, this is a no-op."""
    if frame.implementation == nw.Implementation.POLARS:
        return frame.to_native()
    return frame.collect().to_polars().lazy()


@overload
def switch_implementation_to_polars(frame: DataFrameT) -> DataFrameT: ...


@overload
def switch_implementation_to_polars(frame: LazyFrameT) -> LazyFrameT: ...


def switch_implementation_to_polars(frame: FrameT) -> FrameT:
    if frame.implementation == nw.Implementation.POLARS:
        return frame
    elif isinstance(frame, nw.DataFrame):
        return nw.from_native(frame.to_polars())
    elif isinstance(frame, nw.LazyFrame):
        return nw.from_native(
            frame.collect().to_polars(),
        ).lazy()
    else:
        raise ValueError(f"Unsupported frame type: {type(frame)}")


__all__ = ["collect_to_polars"]
