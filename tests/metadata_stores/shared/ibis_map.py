"""Ibis-backed Map dtype test pack for metadata stores.

Tests that Ibis-backed (lazy) frames are written correctly through the Map
conversion path without materialization.
"""

from __future__ import annotations

from collections.abc import Iterator

import ibis
import narwhals as nw
import pyarrow as pa
import pytest

from metaxy.config import MetaxyConfig
from metaxy.metadata_store import MetadataStore
from metaxy.models.feature_definition import FeatureDefinition


def _collect_to_arrow(frame: nw.LazyFrame | nw.DataFrame) -> pa.Table:
    """Collect a Narwhals frame to a PyArrow Table."""
    return (frame.collect() if isinstance(frame, nw.LazyFrame) else frame).to_arrow()


class IbisMapTests:
    """Tests for Map column handling with Ibis-backed (lazy) frames.

    Verifies that the Ibis path in ``_handle_ibis_map_columns`` stays lazy
    and produces correct results when writing to stores with Map columns.

    Requires ``store`` and ``test_features`` fixtures from the consuming test class.
    """

    @pytest.fixture()
    def polars_map_config(self) -> Iterator[MetaxyConfig]:
        """Activate enable_map_datatype for the duration of the test."""
        config = MetaxyConfig(enable_map_datatype=True)
        with config.use():
            yield config

    @staticmethod
    def _ibis_metadata(
        uids: list[int],
        provenance: list[dict[str, str]],
    ) -> ibis.Table:
        """Build an Ibis memtable with Struct columns."""
        return ibis.memtable({"sample_uid": uids, "metaxy_provenance_by_field": provenance})

    def test_ibis_frame_write_roundtrip(
        self,
        polars_map_config: MetaxyConfig,
        store: MetadataStore,
        test_features: dict[str, FeatureDefinition],
    ) -> None:
        """An Ibis-backed frame with Struct columns round-trips through Map storage."""
        feature = test_features["UpstreamFeatureA"]
        metadata = self._ibis_metadata(
            uids=[1, 2],
            provenance=[{"frames": "h1", "audio": "h2"}, {"frames": "h3", "audio": "h4"}],
        )

        with store.open("w") as s:
            s.write(feature, metadata)

        with store.open("r") as s:
            result = s.read(feature)
            assert result is not None
            table = _collect_to_arrow(result).sort_by("sample_uid")

        assert table.num_rows == 2
        assert table.column("sample_uid").to_pylist() == [1, 2]
        assert pa.types.is_map(table.schema.field("metaxy_provenance_by_field").type)

    def test_ibis_frame_map_values_accessible(
        self,
        polars_map_config: MetaxyConfig,
        store: MetadataStore,
        test_features: dict[str, FeatureDefinition],
    ) -> None:
        """Map values from an Ibis-written frame are accessible after read."""
        feature = test_features["UpstreamFeatureA"]
        metadata = self._ibis_metadata(
            uids=[10, 20],
            provenance=[{"frames": "a1", "audio": "b1"}, {"frames": "a2", "audio": "b2"}],
        )

        with store.open("w") as s:
            s.write(feature, metadata)

        with store.open("r") as s:
            result = s.read(feature)
            assert result is not None
            table = _collect_to_arrow(result).sort_by("sample_uid")

        prov = table.column("metaxy_provenance_by_field")
        assert dict(prov[0].as_py()) == {"frames": "a1", "audio": "b1"}
        assert dict(prov[1].as_py()) == {"frames": "a2", "audio": "b2"}

    def test_ibis_frame_stays_lazy(
        self,
        polars_map_config: MetaxyConfig,
        store: MetadataStore,
        test_features: dict[str, FeatureDefinition],
    ) -> None:
        """An Ibis memtable with Struct columns can be written through the Map path."""
        feature = test_features["UpstreamFeatureA"]
        metadata = self._ibis_metadata(uids=[1], provenance=[{"frames": "h1", "audio": "h2"}])

        with store.open("w") as s:
            s.write(feature, metadata)

        with store.open("r") as s:
            result = s.read(feature)
            assert result is not None
            table = _collect_to_arrow(result)

        assert table.num_rows == 1
        assert table.column("sample_uid").to_pylist() == [1]
        assert pa.types.is_map(table.schema.field("metaxy_provenance_by_field").type)
        assert dict(table.column("metaxy_provenance_by_field")[0].as_py()) == {"frames": "h1", "audio": "h2"}

    def test_ibis_frame_append(
        self,
        polars_map_config: MetaxyConfig,
        store: MetadataStore,
        test_features: dict[str, FeatureDefinition],
    ) -> None:
        """Two Ibis-backed writes with different Map keys produce correct results."""
        feature = test_features["UpstreamFeatureA"]

        batch_1 = self._ibis_metadata(uids=[1], provenance=[{"frames": "f1", "audio": "a1"}])
        batch_2 = self._ibis_metadata(uids=[2], provenance=[{"frames": "f2", "audio": "a2"}])

        with store.open("w") as s:
            s.write(feature, batch_1)
            s.write(feature, batch_2)

        with store.open("r") as s:
            result = s.read(feature)
            assert result is not None
            table = _collect_to_arrow(result).sort_by("sample_uid")

        assert table.num_rows == 2
        assert pa.types.is_map(table.schema.field("metaxy_provenance_by_field").type)
