"""Integration test for the full write-read roundtrip of Arrow Map columns."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import narwhals as nw
import polars as pl
import pyarrow as pa
import pytest
from polars_map import Map

from metaxy.config import MetaxyConfig
from metaxy.utils._arrow_map import (
    convert_extension_maps_to_native,
    convert_maps_to_polars_map,
    convert_structs_to_maps,
)
from metaxy.utils.dataframes import lazy_frame_to_polars
from metaxy.versioning.types import Increment, LazyIncrement

MAP_STR_STR = Map(pl.String(), pl.String())


@pytest.fixture
def provenance_df() -> pl.DataFrame:
    """DataFrame with a struct metaxy_provenance_by_field column (simulating versioning output)."""
    return pl.DataFrame(
        {
            "sample_uid": ["s1", "s2", "s3"],
            "value": [10, 20, 30],
            "metaxy_provenance_by_field": [
                {"default": "abc123", "extra": "def456"},
                {"default": "ghi789", "extra": "jkl012"},
                {"default": "mno345", "extra": "pqr678"},
            ],
        }
    )


class TestMapRoundtrip:
    """Full write-to-read roundtrip through Delta Lake with Arrow Map columns."""

    def test_struct_to_native_arrow_map(self, provenance_df: pl.DataFrame) -> None:
        """Struct → polars_map.Map → Arrow extension → native Arrow MapArray."""
        converted = convert_structs_to_maps(provenance_df, columns=["metaxy_provenance_by_field"])
        arrow_table = convert_extension_maps_to_native(converted.to_arrow())

        map_field = arrow_table.schema.field("metaxy_provenance_by_field")
        assert pa.types.is_map(map_field.type)
        assert pa.types.is_string(map_field.type.key_type)
        assert pa.types.is_string(map_field.type.item_type)

    def test_delta_roundtrip(self, provenance_df: pl.DataFrame, tmp_path: Path) -> None:
        """Write Arrow Map to Delta, read back, reconstruct polars_map.Map column."""
        import deltalake

        # Write path: struct → Map → Arrow → Delta
        converted = convert_structs_to_maps(provenance_df, columns=["metaxy_provenance_by_field"])
        arrow_table = convert_extension_maps_to_native(converted.to_arrow())
        delta_path = str(tmp_path / "test_delta")
        deltalake.write_deltalake(delta_path, arrow_table)

        # Read path: Delta → Polars → polars_map.Map
        read_df = pl.read_delta(delta_path)
        result = convert_maps_to_polars_map(read_df, columns=["metaxy_provenance_by_field"])

        assert result.schema["metaxy_provenance_by_field"] == Map(pl.String(), pl.String())

        defaults = result["metaxy_provenance_by_field"].map.get("default").to_list()  # ty: ignore[unresolved-attribute]
        assert defaults == ["abc123", "ghi789", "mno345"]

        extras = result["metaxy_provenance_by_field"].map.get("extra").to_list()  # ty: ignore[unresolved-attribute]
        assert extras == ["def456", "jkl012", "pqr678"]

    def test_delta_roundtrip_preserves_other_columns(self, provenance_df: pl.DataFrame, tmp_path: Path) -> None:
        """Non-map columns survive the roundtrip unchanged."""
        import deltalake

        converted = convert_structs_to_maps(provenance_df, columns=["metaxy_provenance_by_field"])
        arrow_table = convert_extension_maps_to_native(converted.to_arrow())
        delta_path = str(tmp_path / "test_delta")
        deltalake.write_deltalake(delta_path, arrow_table)

        read_df = pl.read_delta(delta_path)
        result = convert_maps_to_polars_map(read_df, columns=["metaxy_provenance_by_field"])

        assert result["sample_uid"].to_list() == ["s1", "s2", "s3"]
        assert result["value"].to_list() == [10, 20, 30]


# ── Map preservation through conversion helpers ─────────────────────


@pytest.fixture
def polars_map_config() -> Iterator[MetaxyConfig]:
    """Activate enable_map_datatype for the duration of the test."""
    config = MetaxyConfig(enable_map_datatype=True)
    with config.use():
        yield config


@pytest.fixture
def arrow_table_with_map() -> pa.Table:
    """PyArrow table with a native Map column."""
    return pa.table(
        {
            "sample_uid": pa.array([1, 2]),
            "tags": pa.array(
                [
                    [("env", "prod"), ("region", "us")],
                    [("env", "dev"), ("region", "eu")],
                ],
                type=pa.map_(pa.string(), pa.string()),
            ),
        }
    )


class TestMapPreservationInConversions:
    """Map columns survive lazy_frame_to_polars, Increment.to_polars, and LazyIncrement.to_polars."""

    def test_lazy_frame_to_polars_preserves_map_from_pyarrow(
        self,
        polars_map_config: MetaxyConfig,
        arrow_table_with_map: pa.Table,
    ) -> None:
        """PyArrow Map columns become polars_map.Map after lazy_frame_to_polars."""
        nw_lazy = nw.from_native(arrow_table_with_map).lazy()
        result = lazy_frame_to_polars(nw_lazy)

        assert isinstance(result, pl.LazyFrame)
        assert result.collect_schema()["tags"] == MAP_STR_STR

    def test_lazy_frame_to_polars_noop_for_polars_map(
        self,
        polars_map_config: MetaxyConfig,
    ) -> None:
        """Polars-backed LazyFrame with polars_map.Map is returned as-is."""
        tags = pl.Series(
            "tags",
            [[("env", "prod")], [("env", "dev")]],
            dtype=MAP_STR_STR,
        )
        pl_lf = pl.DataFrame({"sample_uid": [1, 2]}).with_columns(tags).lazy()
        nw_lazy = nw.from_native(pl_lf)

        result = lazy_frame_to_polars(nw_lazy)

        assert result is pl_lf
        assert result.collect_schema()["tags"] == MAP_STR_STR

    def test_increment_to_polars_preserves_map_from_pyarrow(
        self,
        polars_map_config: MetaxyConfig,
        arrow_table_with_map: pa.Table,
    ) -> None:
        """Increment.to_polars() preserves Map columns from PyArrow-backed frames."""
        nw_df = nw.from_native(arrow_table_with_map)
        empty = nw.from_native(arrow_table_with_map.slice(0, 0))
        increment = Increment(new=nw_df, stale=empty, orphaned=empty)

        polars_inc = increment.to_polars()

        assert polars_inc.new.schema["tags"] == MAP_STR_STR
        assert polars_inc.new["tags"].map.get("env").to_list() == ["prod", "dev"]  # ty: ignore[unresolved-attribute]

    def test_lazy_increment_to_polars_preserves_map_from_pyarrow(
        self,
        polars_map_config: MetaxyConfig,
        arrow_table_with_map: pa.Table,
    ) -> None:
        """LazyIncrement.to_polars() preserves Map columns from PyArrow-backed frames."""
        nw_lazy = nw.from_native(arrow_table_with_map).lazy()
        empty = nw.from_native(arrow_table_with_map.slice(0, 0)).lazy()
        lazy_inc = LazyIncrement(new=nw_lazy, stale=empty, orphaned=empty)

        polars_inc = lazy_inc.to_polars()

        assert polars_inc.new.collect_schema()["tags"] == MAP_STR_STR
        collected = polars_inc.new.collect()
        assert isinstance(collected, pl.DataFrame)
        assert collected["tags"].map.get("env").to_list() == ["prod", "dev"]  # ty: ignore[unresolved-attribute]
