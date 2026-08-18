"""Tests for schema validation helpers used by MetadataStore."""

from __future__ import annotations

import narwhals as nw
import polars as pl
import pyarrow as pa
from metaxy.utils.dataframes import find_map_columns


class TestFindMapColumns:
    """Tests for find_map_columns — detects narwhals columns of Map type."""

    def test_polars_map_detected(self) -> None:
        """A pl.Map column is detected."""
        s = pl.Series("x", [{"k": "v"}], dtype=pl.Map(pl.String(), pl.String()))
        nw_df = nw.from_native(pl.DataFrame([s]))
        assert find_map_columns(nw_df) == ["x"]

    def test_arrow_map_detected(self) -> None:
        """A native Arrow MapArray column is detected."""
        arrow_table = pa.table({"x": pa.array([[("k", "v")]], type=pa.map_(pa.string(), pa.string()))})
        nw_df = nw.from_native(arrow_table)
        assert find_map_columns(nw_df) == ["x"]

    def test_struct_is_not_map(self) -> None:
        """A Polars Struct column is not Map."""
        nw_df = nw.from_native(pl.DataFrame({"x": [{"a": "1", "b": "2"}]}))
        assert find_map_columns(nw_df) == []

    def test_string_is_not_map(self) -> None:
        """A scalar String column is not Map."""
        nw_df = nw.from_native(pl.DataFrame({"x": ["hello"]}))
        assert find_map_columns(nw_df) == []

    def test_polars_list_of_kv_struct_is_not_map(self) -> None:
        """A Polars List(Struct({key, value})) without pl.Map dtype is not Map."""
        df = pl.DataFrame(
            {"x": [[{"key": "k", "value": "v"}]]},
            schema={"x": pl.List(pl.Struct({"key": pl.String, "value": pl.String}))},
        )
        nw_df = nw.from_native(df)
        assert find_map_columns(nw_df) == []
