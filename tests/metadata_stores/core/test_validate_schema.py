"""Tests for schema validation helpers used by MetadataStore."""

from __future__ import annotations

from collections.abc import Iterator

import narwhals as nw
import polars as pl
import pyarrow as pa
import pytest
from polars_map import Map

from metaxy.config import MetaxyConfig
from metaxy.metadata_store.base import _is_map_column


@pytest.fixture
def polars_map_config() -> Iterator[MetaxyConfig]:
    config = MetaxyConfig(enable_map_datatype=True)
    with config.use():
        yield config


class TestIsMapColumn:
    """Tests for _is_map_column — checks if a narwhals column is a Map type."""

    def test_polars_map_detected(self, polars_map_config: MetaxyConfig) -> None:
        """A polars_map.Map column is detected."""
        s = pl.Series("x", [[("k", "v")]], dtype=Map(pl.String(), pl.String()))
        nw_df = nw.from_native(pl.DataFrame([s]))
        assert _is_map_column(nw_df, "x") is True

    def test_arrow_map_detected(self, polars_map_config: MetaxyConfig) -> None:
        """A native Arrow MapArray column is detected."""
        arrow_table = pa.table({"x": pa.array([[("k", "v")]], type=pa.map_(pa.string(), pa.string()))})
        nw_df = nw.from_native(arrow_table)
        assert _is_map_column(nw_df, "x") is True

    def test_returns_false_without_config(self) -> None:
        """Returns False when enable_map_datatype is not set."""
        s = pl.Series("x", [[("k", "v")]], dtype=Map(pl.String(), pl.String()))
        nw_df = nw.from_native(pl.DataFrame([s]))
        assert _is_map_column(nw_df, "x") is False

    def test_struct_is_not_map(self, polars_map_config: MetaxyConfig) -> None:
        """A Polars Struct column is not Map."""
        nw_df = nw.from_native(pl.DataFrame({"x": [{"a": "1", "b": "2"}]}))
        assert _is_map_column(nw_df, "x") is False

    def test_string_is_not_map(self, polars_map_config: MetaxyConfig) -> None:
        """A scalar String column is not Map."""
        nw_df = nw.from_native(pl.DataFrame({"x": ["hello"]}))
        assert _is_map_column(nw_df, "x") is False

    def test_polars_list_of_kv_struct_is_not_map(self, polars_map_config: MetaxyConfig) -> None:
        """A Polars List(Struct({key, value})) without polars_map.Map dtype is not Map."""
        df = pl.DataFrame(
            {"x": [[{"key": "k", "value": "v"}]]},
            schema={"x": pl.List(pl.Struct({"key": pl.String, "value": pl.String}))},
        )
        nw_df = nw.from_native(df)
        assert _is_map_column(nw_df, "x") is False
