"""Unit tests for Arrow Map conversion utilities."""

from __future__ import annotations

import polars as pl
import pyarrow as pa
import pytest
from metaxy.utils._arrow_map import convert_structs_to_maps

MAP_STR_STR = pl.Map(pl.String(), pl.String())


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


# ── convert_structs_to_maps ──────────────────────────────────────────


class TestConvertStructsToMaps:
    """Tests for the write-path Struct to pl.Map conversion."""

    def test_struct_to_map_basic(self, struct_df: pl.DataFrame) -> None:
        """Struct column becomes pl.Map, values accessible via .map.get()."""
        result = convert_structs_to_maps(struct_df, columns=["metaxy_provenance_by_field"])

        assert result.schema["metaxy_provenance_by_field"] == MAP_STR_STR
        assert result["metaxy_provenance_by_field"].map.get("alpha").to_list() == ["a1", "a2"]
        assert result["metaxy_provenance_by_field"].map.get("beta").to_list() == ["b1", "b2"]

    def test_roundtrip_to_native_arrow_map(self, struct_df: pl.DataFrame) -> None:
        """Struct → pl.Map → native Arrow MapArray."""
        native = convert_structs_to_maps(struct_df, columns=["metaxy_provenance_by_field"]).to_arrow()

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
