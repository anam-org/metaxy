from typing import overload

import narwhals as nw
import polars as pl
from narwhals.typing import DataFrameT, Frame, FrameT, LazyFrameT


def collect_to_polars(frame: Frame) -> pl.DataFrame:
    """Helper to convert a Narwhals frame into an eager Polars DataFrame."""

    return frame.lazy().collect().to_polars()


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
