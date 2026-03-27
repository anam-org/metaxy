"""Map dtype test pack for metadata stores.

Tests that stores correctly handle polars-map Map columns on both read and write,
for both metaxy-managed (*_by_field) and user-defined Map columns.
"""

from __future__ import annotations

from collections.abc import Iterator

import polars as pl
import pyarrow as pa
import pytest
from polars_map import Map

from metaxy._utils import collect_to_polars
from metaxy.config import MetaxyConfig
from metaxy.metadata_store import MetadataStore
from metaxy.models.feature_definition import FeatureDefinition

MAP_STR_STR = Map(pl.String(), pl.String())


class MapDtypeTests:
    """Tests for Map column support in metadata stores.

    Requires ``store`` and ``test_features`` fixtures from the consuming test class.
    """

    @pytest.fixture
    def polars_map_config(self) -> Iterator[MetaxyConfig]:
        """Activate enable_map_datatype for the duration of the test."""
        config = MetaxyConfig(enable_map_datatype=True)
        with config.use():
            yield config

    # ── enable_map_datatype: Struct → Map on write, Map on read ──────────

    def test_write_stores_native_map_type(
        self,
        polars_map_config: MetaxyConfig,
        store: MetadataStore,
        test_features: dict[str, FeatureDefinition],
    ) -> None:
        """With enable_map_datatype, metaxy *_by_field Struct columns are stored as native Map."""
        feature = test_features["UpstreamFeatureA"]
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {"frames": "h1", "audio": "h2"},
                    {"frames": "h3", "audio": "h4"},
                ],
            }
        )

        with store.open("w") as s:
            s.write(feature, metadata)

        with store.open("r") as s:
            result = s.read(feature)
            assert result is not None
            df = collect_to_polars(result)

        assert df.schema["metaxy_provenance_by_field"] == MAP_STR_STR
        df = df.sort("sample_uid")
        frames = df["metaxy_provenance_by_field"].map.get("frames").to_list()  # ty: ignore[unresolved-attribute]
        audio = df["metaxy_provenance_by_field"].map.get("audio").to_list()  # ty: ignore[unresolved-attribute]
        assert frames == ["h1", "h3"]
        assert audio == ["h2", "h4"]

    def test_read_map_values_accessible(
        self,
        polars_map_config: MetaxyConfig,
        store: MetadataStore,
        test_features: dict[str, FeatureDefinition],
    ) -> None:
        """Map column values are accessible via .map.get() after read."""
        feature = test_features["UpstreamFeatureA"]
        metadata = pl.DataFrame(
            {
                "sample_uid": [10, 20, 30],
                "metaxy_provenance_by_field": [
                    {"frames": "a1", "audio": "b1"},
                    {"frames": "a2", "audio": "b2"},
                    {"frames": "a3", "audio": "b3"},
                ],
            }
        )

        with store.open("w") as s:
            s.write(feature, metadata)

        with store.open("r") as s:
            result = s.read(feature)
            assert result is not None
            df = collect_to_polars(result).sort("sample_uid")

        frames = df["metaxy_provenance_by_field"].map.get("frames").to_list()  # ty: ignore[unresolved-attribute]
        audio = df["metaxy_provenance_by_field"].map.get("audio").to_list()  # ty: ignore[unresolved-attribute]
        assert frames == ["a1", "a2", "a3"]
        assert audio == ["b1", "b2", "b3"]

    # ── User-defined Map columns ───────────────────────────────────────

    def test_user_map_column_roundtrip(
        self,
        polars_map_config: MetaxyConfig,
        store: MetadataStore,
        test_features: dict[str, FeatureDefinition],
    ) -> None:
        """User-defined Map columns survive write→read roundtrip with enable_map_datatype."""
        feature = test_features["UpstreamFeatureA"]
        user_map = pl.Series(
            "tags",
            [[("env", "prod"), ("region", "us")], [("env", "dev"), ("region", "eu")]],
            dtype=MAP_STR_STR,
        )
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {"frames": "h1", "audio": "h2"},
                    {"frames": "h3", "audio": "h4"},
                ],
            }
        ).with_columns(user_map)

        with store.open("w") as s:
            s.write(feature, metadata)

        with store.open("r") as s:
            result = s.read(feature)
            assert result is not None
            df = collect_to_polars(result).sort("sample_uid")

        assert df.schema["tags"] == MAP_STR_STR
        envs = df["tags"].map.get("env").to_list()  # ty: ignore[unresolved-attribute]
        assert envs == ["prod", "dev"]

    # ── Write accepts pre-built Map columns ────────────────────────────

    def test_write_accepts_map_provenance_column(
        self,
        polars_map_config: MetaxyConfig,
        store: MetadataStore,
        test_features: dict[str, FeatureDefinition],
    ) -> None:
        """Writing a DataFrame where metaxy_provenance_by_field is already Map works."""
        feature = test_features["UpstreamFeatureA"]
        prov_map = pl.Series(
            "metaxy_provenance_by_field",
            [[("frames", "h1"), ("audio", "h2")]],
            dtype=MAP_STR_STR,
        )
        metadata = pl.DataFrame({"sample_uid": [1]}).with_columns(prov_map)

        with store.open("w") as s:
            s.write(feature, metadata)

        with store.open("r") as s:
            result = s.read(feature)
            assert result is not None
            df = collect_to_polars(result)

        assert df.schema["metaxy_provenance_by_field"] == MAP_STR_STR
        frames = df["metaxy_provenance_by_field"].map.get("frames").to_list()  # ty: ignore[unresolved-attribute]
        audio = df["metaxy_provenance_by_field"].map.get("audio").to_list()  # ty: ignore[unresolved-attribute]
        assert frames == ["h1"]
        assert audio == ["h2"]

    def test_non_map_columns_unchanged(
        self,
        polars_map_config: MetaxyConfig,
        store: MetadataStore,
        test_features: dict[str, FeatureDefinition],
    ) -> None:
        """Scalar columns are unaffected by Map conversions."""
        feature = test_features["UpstreamFeatureA"]
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h2"}],
            }
        )

        with store.open("w") as s:
            s.write(feature, metadata)

        with store.open("r") as s:
            result = s.read(feature)
            assert result is not None
            df = collect_to_polars(result)

        assert df["sample_uid"].to_list() == [1]

    def test_append_with_different_map_keys(
        self,
        polars_map_config: MetaxyConfig,
        store: MetadataStore,
        test_features: dict[str, FeatureDefinition],
    ) -> None:
        """Two writes with different Map keys should both be readable."""
        feature = test_features["UpstreamFeatureA"]

        batch_1 = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "f1", "audio": "a1"}],
            }
        )
        batch_2 = pl.DataFrame(
            {
                "sample_uid": [2],
                "metaxy_provenance_by_field": [{"frames": "f2", "audio": "a2"}],
            }
        )

        with store.open("w") as s:
            s.write(feature, batch_1)
            s.write(feature, batch_2)

        with store.open("r") as s:
            result = s.read(feature)
            assert result is not None
            df = collect_to_polars(result).sort("sample_uid")

        assert df.schema["metaxy_provenance_by_field"] == MAP_STR_STR
        assert df.height == 2
        frames = df["metaxy_provenance_by_field"].map.get("frames").to_list()  # ty: ignore[unresolved-attribute]
        audio = df["metaxy_provenance_by_field"].map.get("audio").to_list()  # ty: ignore[unresolved-attribute]
        assert frames == ["f1", "f2"]
        assert audio == ["a1", "a2"]

    # ── Arrow table with Map columns ───────────────────────────────────

    def test_write_arrow_table_with_map_columns(
        self,
        polars_map_config: MetaxyConfig,
        store: MetadataStore,
        test_features: dict[str, FeatureDefinition],
    ) -> None:
        """Writing a pa.Table with native Map columns for both metaxy and user columns."""
        feature = test_features["UpstreamFeatureA"]

        arrow_table = pa.table(
            {
                "sample_uid": pa.array([1, 2]),
                "metaxy_provenance_by_field": pa.array(
                    [
                        [("frames", "f1"), ("audio", "a1")],
                        [("frames", "f2"), ("audio", "a2")],
                    ],
                    type=pa.map_(pa.string(), pa.string()),
                ),
                "user_tags": pa.array(
                    [
                        [("env", "prod")],
                        [("env", "dev")],
                    ],
                    type=pa.map_(pa.string(), pa.string()),
                ),
            }
        )

        with store.open("w") as s:
            s.write(feature, arrow_table)

        with store.open("r") as s:
            result = s.read(feature)
            assert result is not None
            df = collect_to_polars(result).sort("sample_uid")

        assert df.schema["metaxy_provenance_by_field"] == MAP_STR_STR
        assert df.schema["user_tags"] == MAP_STR_STR

        frames = df["metaxy_provenance_by_field"].map.get("frames").to_list()  # ty: ignore[unresolved-attribute]
        assert frames == ["f1", "f2"]

        envs = df["user_tags"].map.get("env").to_list()  # ty: ignore[unresolved-attribute]
        assert envs == ["prod", "dev"]

    # ── User struct columns are not converted ──────────────────────────

    def test_user_struct_columns_not_converted_to_map(
        self,
        polars_map_config: MetaxyConfig,
        store: MetadataStore,
        test_features: dict[str, FeatureDefinition],
    ) -> None:
        """User-defined Struct columns are preserved as Struct, not converted to Map."""
        feature = test_features["UpstreamFeatureA"]
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h2"}],
                "user_struct": [{"x": "1", "y": "2"}],
            }
        )

        with store.open("w") as s:
            s.write(feature, metadata)

        with store.open("r") as s:
            result = s.read(feature)
            assert result is not None
            df = collect_to_polars(result)

        # metaxy column should be Map
        assert df.schema["metaxy_provenance_by_field"] == MAP_STR_STR
        # user struct column should remain Struct
        assert isinstance(df.schema["user_struct"], pl.Struct)

    # ── Non-string value types ───────────────────────────────────────

    def test_user_map_with_non_string_types(
        self,
        polars_map_config: MetaxyConfig,
        store: MetadataStore,
        test_features: dict[str, FeatureDefinition],
    ) -> None:
        """User-defined Map(Int32, Float64) columns survive write→read roundtrip."""
        feature = test_features["UpstreamFeatureA"]
        map_int_float = Map(pl.Int32(), pl.Float64())
        scores = pl.Series(
            "scores",
            [[(1, 0.95), (2, 0.87)], [(1, 0.91), (2, 0.82)]],
            dtype=map_int_float,
        )
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {"frames": "h1", "audio": "h2"},
                    {"frames": "h3", "audio": "h4"},
                ],
            }
        ).with_columns(scores)

        with store.open("w") as s:
            s.write(feature, metadata)

        with store.open("r") as s:
            result = s.read(feature)
            assert result is not None
            df = collect_to_polars(result).sort("sample_uid")

        assert df.schema["scores"] == map_int_float
        assert df["scores"].map.get(1).to_list() == [0.95, 0.91]  # ty: ignore[unresolved-attribute]
        assert df["scores"].map.get(2).to_list() == [0.87, 0.82]  # ty: ignore[unresolved-attribute]
