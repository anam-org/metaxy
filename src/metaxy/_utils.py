from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast, overload

import narwhals as nw
import polars as pl
from narwhals.typing import DataFrameT, Frame, FrameT, LazyFrameT

if TYPE_CHECKING:
    pass


def collect_to_polars(frame: Frame) -> pl.DataFrame:
    """Helper to convert a Narwhals frame into an eager Polars DataFrame."""

    return frame.lazy().collect().to_polars()


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


def collect_batches(
    df: Frame,
    chunk_size: int | None = None,
    order_by: Sequence[str] | None = None,
    **kwargs,
) -> Iterator[nw.DataFrame[Any]]:
    """
    Collect batches of data from a DataFrame or LazyFrame.

    Parameters:
        df: The frame to collect batches from.
        chunk_size: The size of each batch. If None, collect everything at once.
        order_by: The columns the dataframe is supposed to be ordered by. The user is responsible for ensuring the dataframe is sorted by these columns. If `None`, no ordering is expected.
        **kwargs: Additional keyword arguments to pass to the backend-specific collect method.

    Yields:
        An iterator over the collected batches. Batches are collected sequentially.
    """
    if isinstance(df, nw.DataFrame):
        df = df.lazy()

    assert isinstance(df, nw.LazyFrame)

    if df.implementation == nw.Implementation.POLARS:
        df_polars = cast(pl.LazyFrame, df.to_native())

        if chunk_size is None:
            # No chunking requested, collect everything at once
            yield nw.from_native(df_polars.collect(**kwargs))
        else:
            if order_by is not None:
                # the user requests a specific ordering
                # and we assume the DataFrame has already been sorted by these columns
                # so all we need to do is not to break it
                kwargs["maintain_order"] = True

            for batch in df_polars.collect_batches(chunk_size=chunk_size, **kwargs):
                yield nw.from_native(batch)

    # other backends don't have a native implementation for this operations,
    # so we have to think about ordering
    # the use must provide a valid order_by argument
    # and actually order the table in advance (or maybe we can do it automatically, not sure?)
    else:
        if chunk_size is None:
            yield df.collect()
        else:
            drop_cols = ["__collect__batches_index__"]
            if order_by is None:
                # there is no ordering requested
                # but with_row_index needs it
                # so we'll just make it up with minimal performance overhead
                order_by = ["__collect_batches_order__"]
                df = df.with_columns(nw.lit(True).alias("__collect_batches_order__"))
                drop_cols.append("__collect_batches_order__")

            df = df.with_row_index("__collect__batches_index__", order_by=order_by)

            total_len = df.select(nw.len()).collect().item()

            for start in range(0, total_len, chunk_size):
                yield (
                    df.filter(
                        (nw.col("__collect__batches_index__") >= start)
                        & (nw.col("__collect__batches_index__") < start + chunk_size)
                    )
                    .drop(*drop_cols)
                    .collect()
                )


__all__ = ["collect_to_polars", "collect_batches"]
