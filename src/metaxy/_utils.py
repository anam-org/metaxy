import polars as pl
from narwhals.typing import Frame


def collect_to_polars(frame: Frame) -> pl.DataFrame:
    """Helper to convert a Narwhals frame into an eager Polars DataFrame."""

    return frame.lazy().collect().to_polars()


__all__ = ["collect_to_polars"]
