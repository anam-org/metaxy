import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from metaxy._utils import collect_to_polars


class _FakeLazyFrame:
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def collect(self) -> pl.DataFrame:
        return self._df


class _FakeFrame:
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def lazy(self) -> _FakeLazyFrame:
        return _FakeLazyFrame(self._df)


def test_collect_to_polars_native_polars_dataframe() -> None:
    df = pl.DataFrame({"id": [1, 2], "value": ["a", "b"]})

    result = collect_to_polars(df)

    assert result.equals(df)


def test_collect_to_polars_native_polars_lazyframe() -> None:
    lazy_df = pl.DataFrame({"id": [1, 2]}).lazy()

    result = collect_to_polars(lazy_df)

    assert isinstance(result, pl.DataFrame)
    assert result["id"].to_list() == [1, 2]


def test_collect_to_polars_narwhals_polars_dataframe() -> None:
    df = pl.DataFrame({"id": [1, 2], "value": ["a", "b"]})
    nw_df = nw.from_native(df)

    result = collect_to_polars(nw_df)

    assert result.equals(df)


def test_collect_to_polars_narwhals_non_polars_dataframe() -> None:
    df = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
    nw_df = nw.from_native(df)

    result = collect_to_polars(nw_df)

    assert isinstance(result, pl.DataFrame)
    assert result["id"].to_list() == [1, 2]
    assert result["value"].to_list() == ["a", "b"]


def test_collect_to_polars_narwhals_polars_lazyframe() -> None:
    lazy_df = pl.DataFrame({"id": [1, 2]}).lazy()
    nw_lazy = nw.from_native(lazy_df, eager_only=False)

    result = collect_to_polars(nw_lazy)

    assert isinstance(result, pl.DataFrame)
    assert result["id"].to_list() == [1, 2]


def test_collect_to_polars_narwhals_non_polars_lazyframe() -> None:
    ibis = pytest.importorskip("ibis")

    conn = ibis.duckdb.connect(":memory:")
    source_df = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
    conn.create_table("tmp_collect_to_polars", obj=source_df, overwrite=True)
    table = conn.table("tmp_collect_to_polars")
    nw_lazy = nw.from_native(table, eager_only=False)

    result = collect_to_polars(nw_lazy)

    assert isinstance(result, pl.DataFrame)
    assert result["id"].to_list() == [1, 2]
    assert result["value"].to_list() == ["a", "b"]


def test_collect_to_polars_fallback_collect_returns_polars_dataframe() -> None:
    df = pl.DataFrame({"id": [1], "name": ["x"]})
    frame = _FakeFrame(df)

    result = collect_to_polars(frame)  # ty: ignore[invalid-argument-type]

    assert result.equals(df)
