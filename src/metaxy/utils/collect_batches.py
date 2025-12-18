from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

import narwhals as nw
import polars as pl
import pyarrow as pa
from narwhals.typing import Frame

if TYPE_CHECKING:
    import ibis.expr.types as ir


def collect_batches(
    df: Frame,
    chunk_size: int | None = None,
    **kwargs: Any,
) -> Iterator[nw.DataFrame[Any]]:
    """
    Collect batches of data from a DataFrame or LazyFrame.

    Uses native batch iteration when available (Polars, Ibis) to avoid
    recomputation overhead.

    Parameters:
        df: The frame to collect batches from.
        chunk_size: The size of each batch. If None, collect everything at once.
        **kwargs: Additional keyword arguments to pass to the backend-specific collect method.

    Yields:
        An iterator over the collected batches.
    """
    if isinstance(df, nw.DataFrame):
        df = df.lazy()

    assert isinstance(df, nw.LazyFrame)

    if df.implementation == nw.Implementation.POLARS:
        yield from _collect_batches_polars(df, chunk_size, **kwargs)
    elif df.implementation == nw.Implementation.IBIS:
        yield from _collect_batches_ibis(df, chunk_size, **kwargs)
    else:
        raise NotImplementedError(
            f"collect_batches is not supported for {df.implementation}. Supported backends: Polars, Ibis."
        )


def _collect_batches_polars(
    df: nw.LazyFrame[Any],
    chunk_size: int | None,
    **kwargs: Any,
) -> Iterator[nw.DataFrame[Any]]:
    """Collect batches using Polars native collect_batches."""
    df_polars = cast(pl.LazyFrame, df.to_native())

    if chunk_size is None:
        yield nw.from_native(df_polars.collect(**kwargs))
    else:
        for batch in df_polars.collect_batches(chunk_size=chunk_size, **kwargs):
            yield nw.from_native(batch)


def _collect_batches_ibis(
    df: nw.LazyFrame[Any],
    chunk_size: int | None,
    **kwargs: Any,
) -> Iterator[nw.DataFrame[Any]]:
    """Collect batches using Ibis native to_pyarrow_batches."""
    ibis_table = cast("ir.Table", df.to_native())

    if chunk_size is None:
        yield df.collect()
    else:
        batch_reader = ibis_table.to_pyarrow_batches(chunk_size=chunk_size, **kwargs)
        for batch in batch_reader:
            yield nw.from_native(pa.Table.from_batches([batch]))


__all__ = ["collect_batches"]
