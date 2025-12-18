"""Test that PerformanceWarning from Polars lazy frames is caught by pytest."""

import narwhals as nw
import polars as pl
import pytest


@pytest.mark.xfail(
    raises=pl.exceptions.PerformanceWarning,
    reason="Accessing .columns on a LazyFrame triggers PerformanceWarning",
    strict=True,
)
def test_lazyframe_columns_raises_performance_warning() -> None:
    """Verify that accessing .columns on a narwhals LazyFrame raises PerformanceWarning.

    This test is expected to fail (xfail) because our pytest configuration
    treats PerformanceWarning as an error. If this test passes instead of
    failing, it means the warning is not being raised or caught properly.
    """
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    nw_lf = nw.from_native(lf)

    # This should trigger PerformanceWarning
    _ = nw_lf.columns
