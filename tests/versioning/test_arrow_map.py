"""Unit tests for Arrow Map conversion utilities."""

from __future__ import annotations

import polars as pl
import pyarrow as pa
import pytest
from polars_map import Map

from metaxy.versioning._arrow_map import (
    convert_extension_maps_to_native,
    convert_maps_to_polars_map,
    convert_structs_to_maps,
)

MAP_STR_STR = Map(pl.String(), pl.String())


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def struct_df() -> pl.DataFrame:
    """Polars DataFrame with a Struct column."""
    return pl.DataFrame(
        {
            "sample_uid": ["s1", "s2"],
            "metaxy_provenance_by_field": [
                {"alpha": "a1", "beta": "b1"},
                {"alpha": "a2", "beta": "b2"},
            ],
        }
    )


@pytest.fixture
def multi_column_df() -> pl.DataFrame:
    """Polars DataFrame with metaxy struct columns and non-metaxy columns."""
    return pl.DataFrame(
        {
            "sample_uid": ["s1"],
            "metaxy_data_version_by_field": [{"x": "1", "y": "2"}],
            "other_struct": [{"x": "3", "y": "4"}],
            "score": [0.9],
        }
    )


@pytest.fixture
def list_kv_df() -> pl.DataFrame:
    """Polars DataFrame with a List(Struct({key, value})) column (as Arrow Map appears in Polars)."""
    return pl.DataFrame(
        {
            "sample_uid": ["s1", "s2"],
            "metaxy_provenance_by_field": [
                [{"key": "alpha", "value": "a1"}, {"key": "beta", "value": "b1"}],
                [{"key": "alpha", "value": "a2"}, {"key": "beta", "value": "b2"}],
            ],
        }
    )


# ── convert_structs_to_maps ──────────────────────────────────────────


class TestConvertStructsToMaps:
    """Tests for the write-path Struct to polars_map.Map conversion."""

    def test_struct_to_map_basic(self, struct_df: pl.DataFrame) -> None:
        """Struct column becomes polars_map.Map, values accessible via .map.get()."""
        result = convert_structs_to_maps(struct_df, columns=["metaxy_provenance_by_field"])

        assert result.schema["metaxy_provenance_by_field"] == MAP_STR_STR
        assert result["metaxy_provenance_by_field"].map.get("alpha").to_list() == ["a1", "a2"]  # ty: ignore[unresolved-attribute]
        assert result["metaxy_provenance_by_field"].map.get("beta").to_list() == ["b1", "b2"]  # ty: ignore[unresolved-attribute]

    def test_roundtrip_to_native_arrow_map(self, struct_df: pl.DataFrame) -> None:
        """Struct → polars_map.Map → Arrow extension → native Arrow MapArray."""
        converted = convert_structs_to_maps(struct_df, columns=["metaxy_provenance_by_field"])
        arrow_table = converted.to_arrow()
        native = convert_extension_maps_to_native(arrow_table)

        map_field = native.schema.field("metaxy_provenance_by_field")
        assert pa.types.is_map(map_field.type)

        row0 = native.column("metaxy_provenance_by_field")[0].as_py()
        assert row0 == [("alpha", "a1"), ("beta", "b1")]

    def test_only_specified_columns_converted(self, multi_column_df: pl.DataFrame) -> None:
        """Only specified columns are converted, others remain unchanged."""
        result = convert_structs_to_maps(multi_column_df, columns=["metaxy_data_version_by_field"])

        assert result.schema["metaxy_data_version_by_field"] == MAP_STR_STR
        assert isinstance(result.schema["other_struct"], pl.Struct)
        assert result.schema["score"] == pl.Float64

    def test_explicit_columns_parameter(self, multi_column_df: pl.DataFrame) -> None:
        """Explicit columns parameter targets only specified columns."""
        result = convert_structs_to_maps(multi_column_df, columns=["other_struct"])

        assert result.schema["other_struct"] == MAP_STR_STR
        assert isinstance(result.schema["metaxy_data_version_by_field"], pl.Struct)

    def test_non_struct_columns_unchanged(self, struct_df: pl.DataFrame) -> None:
        """Non-struct columns are left unchanged."""
        result = convert_structs_to_maps(struct_df, columns=["metaxy_provenance_by_field"])

        assert result.schema["sample_uid"] == pl.String
        assert result["sample_uid"].to_list() == ["s1", "s2"]

    def test_empty_dataframe(self) -> None:
        """Empty DataFrame with struct schema produces empty map column."""
        empty = pl.DataFrame(
            schema={
                "id": pl.String,
                "metaxy_provenance_by_field": pl.Struct({"a": pl.String}),
            }
        )
        result = convert_structs_to_maps(empty, columns=["metaxy_provenance_by_field"])

        assert result.schema["metaxy_provenance_by_field"] == MAP_STR_STR
        assert result.height == 0


# ── convert_maps_to_polars_map ───────────────────────────────────────


class TestConvertMapsToPolarsMap:
    """Tests for the read-path List(Struct({key, value})) to polars_map.Map conversion."""

    def test_dataframe_to_polars_map(self, list_kv_df: pl.DataFrame) -> None:
        """DataFrame with List(Struct({key, value})) column becomes polars_map.Map dtype."""
        result = convert_maps_to_polars_map(list_kv_df, columns=["metaxy_provenance_by_field"])

        assert isinstance(result, pl.DataFrame)
        assert result.schema["metaxy_provenance_by_field"] == MAP_STR_STR

        vals = result["metaxy_provenance_by_field"].map.get("alpha").to_list()  # ty: ignore[unresolved-attribute]
        assert vals == ["a1", "a2"]

    def test_lazyframe_preserves_type(self, list_kv_df: pl.DataFrame) -> None:
        """LazyFrame input produces LazyFrame output."""
        lazy = list_kv_df.lazy()
        result = convert_maps_to_polars_map(lazy, columns=["metaxy_provenance_by_field"])

        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert collected.schema["metaxy_provenance_by_field"] == MAP_STR_STR  # ty: ignore[unresolved-attribute]

    def test_explicit_columns_parameter(self, list_kv_df: pl.DataFrame) -> None:
        """Explicit columns parameter targets only specified columns."""
        df = list_kv_df.with_columns(
            pl.col("metaxy_provenance_by_field").alias("other_map_col"),
        )
        result = convert_maps_to_polars_map(df, columns=["metaxy_provenance_by_field"])

        assert result.schema["metaxy_provenance_by_field"] == MAP_STR_STR
        assert result.schema["other_map_col"] != MAP_STR_STR

    def test_non_matching_columns_skipped(self) -> None:
        """Columns that are not List(Struct({key, value})) are skipped."""
        df = pl.DataFrame(
            {
                "name": ["hello", "world"],
                "score": [1.0, 2.0],
            }
        )
        result = convert_maps_to_polars_map(df, columns=["name", "score"])

        assert result.schema == df.schema
