import narwhals as nw
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from metaxy.utils import collect_to_arrow


def test_polars_dataframe() -> None:
    df = pl.DataFrame({"id": [1, 2], "value": ["a", "b"]})

    result = collect_to_arrow(df)

    assert isinstance(result, pa.Table)
    assert result.column("id").to_pylist() == [1, 2]
    assert result.column("value").to_pylist() == ["a", "b"]


def test_polars_lazyframe() -> None:
    result = collect_to_arrow(pl.DataFrame({"id": [1, 2]}).lazy())

    assert isinstance(result, pa.Table)
    assert result.column("id").to_pylist() == [1, 2]


def test_narwhals_polars_dataframe() -> None:
    df = pl.DataFrame({"id": [1, 2], "value": ["a", "b"]})

    result = collect_to_arrow(nw.from_native(df))

    assert isinstance(result, pa.Table)
    assert result.column("id").to_pylist() == [1, 2]


def test_narwhals_polars_lazyframe() -> None:
    nw_lazy = nw.from_native(pl.DataFrame({"id": [1, 2]}).lazy(), eager_only=False)

    result = collect_to_arrow(nw_lazy)

    assert isinstance(result, pa.Table)
    assert result.column("id").to_pylist() == [1, 2]


def test_narwhals_pandas_dataframe() -> None:
    nw_df = nw.from_native(pd.DataFrame({"id": [1, 2], "value": ["a", "b"]}))

    result = collect_to_arrow(nw_df)

    assert isinstance(result, pa.Table)
    assert result.column("id").to_pylist() == [1, 2]
    assert result.column("value").to_pylist() == ["a", "b"]


@pytest.fixture
def polars_map_df() -> pl.DataFrame:
    """Build a Polars DataFrame with a polars_map.Map column from a native Arrow map."""
    pytest.importorskip("polars_map")
    from metaxy.versioning._arrow_map import convert_maps_to_polars_map

    arrow_table = pa.table(
        {
            "id": [1, 2],
            "tags": pa.array(
                [[("a", "x"), ("b", "y")], [("c", "z")]],
                type=pa.map_(pa.string(), pa.string()),
            ),
        }
    )
    df: pl.DataFrame = pl.from_arrow(arrow_table)  # ty: ignore[invalid-assignment]
    return convert_maps_to_polars_map(df, columns=["tags"])


def test_polars_map_columns_become_native_arrow_map(polars_map_df: pl.DataFrame) -> None:
    result = collect_to_arrow(polars_map_df)

    assert pa.types.is_map(result.schema.field("tags").type)
    assert dict(result.column("tags")[0].as_py()) == {"a": "x", "b": "y"}
    assert dict(result.column("tags")[1].as_py()) == {"c": "z"}


def test_polars_map_columns_from_lazyframe(polars_map_df: pl.DataFrame) -> None:
    result = collect_to_arrow(polars_map_df.lazy())

    assert pa.types.is_map(result.schema.field("tags").type)
    assert dict(result.column("tags")[0].as_py()) == {"a": "x", "b": "y"}


def test_non_polars_map_columns_unaffected() -> None:
    """Arrow map columns from non-Polars backends stay as-is (already native)."""
    arrays = [
        pa.array([1, 2]),
        pa.array([[("a", "x")], [("b", "y")]], type=pa.map_(pa.string(), pa.string())),
    ]
    table = pa.table({"id": arrays[0], "tags": arrays[1]})
    nw_df = nw.from_native(table)

    result = collect_to_arrow(nw_df)

    assert pa.types.is_map(result.schema.field("tags").type)
    assert dict(result.column("tags")[0].as_py()) == {"a": "x"}
