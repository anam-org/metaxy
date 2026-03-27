"""Integration test for the full write-read roundtrip of Arrow Map columns."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest
from polars_map import Map

from metaxy.versioning._arrow_map import (
    convert_extension_maps_to_native,
    convert_maps_to_polars_map,
    convert_structs_to_maps,
)


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
