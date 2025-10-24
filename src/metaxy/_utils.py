import narwhals as nw
import polars as pl


def collect_to_polars(lazy_frame: nw.LazyFrame) -> pl.DataFrame:
    """Helper to collect a Narwhals LazyFrame and convert to Polars DataFrame.

    This handles all backend conversions (Polars, DuckDB/PyArrow, etc.) transparently.
    Used throughout tests for materializing query results.
    """
    return lazy_frame.collect().to_polars()
