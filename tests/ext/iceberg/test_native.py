"""Apache Iceberg metadata store tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from packaging.version import Version

from metaxy import HashAlgorithm
from metaxy.ext.polars.handlers.iceberg import IcebergMetadataStore
from metaxy.metadata_store import MetadataStore
from metaxy.models.feature_definition import FeatureDefinition
from metaxy.models.types import FeatureKey
from tests.metadata_stores.shared import (
    CRUDTests,
    DeletionTests,
    DisplayTests,
    FilterTests,
    MapDtypeTests,
    ResolveUpdateTests,
    VersioningTests,
    WriteTests,
)


@pytest.fixture
def iceberg_store(tmp_path: Path) -> IcebergMetadataStore:
    """Default Iceberg store with warehouse at tmp_path/iceberg_store."""
    return IcebergMetadataStore(warehouse=tmp_path / "iceberg_store")


@pytest.fixture
def iceberg_custom_ns_store(tmp_path: Path) -> IcebergMetadataStore:
    """Iceberg store with a custom namespace."""
    return IcebergMetadataStore(warehouse=tmp_path / "iceberg_store", namespace="custom_ns")


@pytest.mark.iceberg
@pytest.mark.polars
class TestIceberg(
    CRUDTests,
    DeletionTests,
    DisplayTests,
    FilterTests,
    MapDtypeTests,
    ResolveUpdateTests,
    VersioningTests,
    WriteTests,
):
    @pytest.fixture
    def store(self, tmp_path: Path) -> MetadataStore:
        return IcebergMetadataStore(
            warehouse=tmp_path / "iceberg_store",
            hash_algorithm=HashAlgorithm.XXHASH64,
        )


def test_iceberg_table_identifier(iceberg_store: IcebergMetadataStore) -> None:
    """Verify _table_identifier maps feature keys to (namespace, table_name) tuples."""
    assert iceberg_store._table_identifier(FeatureKey("feature")) == ("metaxy", "feature")
    assert iceberg_store._table_identifier(FeatureKey("a/b/c")) == ("metaxy", "a__b__c")
    assert iceberg_store._table_identifier(FeatureKey("my_feature/v1")) == ("metaxy", "my_feature__v1")


def test_iceberg_schema_evolution(
    iceberg_store: IcebergMetadataStore, test_features: dict[str, FeatureDefinition]
) -> None:
    """Verify schema evolution when writing with new columns."""
    feature_cls = test_features["UpstreamFeatureA"]

    with iceberg_store.open("w") as store:
        store.write(
            feature_cls,
            pl.DataFrame(
                {
                    "sample_uid": [1],
                    "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
                }
            ),
        )

        store.write(
            feature_cls,
            pl.DataFrame(
                {
                    "sample_uid": [2],
                    "metaxy_provenance_by_field": [{"frames": "h2", "audio": "h2"}],
                    "extra_col": ["new"],
                }
            ),
        )

        result = store.read(feature_cls)
        assert result is not None
        collected = result.collect().to_native()
        assert collected.height == 2
        assert "extra_col" in collected.columns


def test_iceberg_custom_namespace(
    iceberg_custom_ns_store: IcebergMetadataStore, test_features: dict[str, FeatureDefinition]
) -> None:
    """Verify custom namespace is used for table identifiers."""
    feature_cls = test_features["UpstreamFeatureA"]

    with iceberg_custom_ns_store.open("w") as store:
        store.write(
            feature_cls,
            pl.DataFrame(
                {
                    "sample_uid": [1],
                    "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
                }
            ),
        )

        assert store.has_feature(feature_cls, check_fallback=False)
        assert store.catalog.list_tables("custom_ns")
        assert ("metaxy",) not in store.catalog.list_namespaces()


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.39.0"),
    reason="sink_iceberg requires Polars >= 1.39.0",
)
def test_iceberg_sink_lazyframe(
    iceberg_store: IcebergMetadataStore, test_features: dict[str, FeatureDefinition]
) -> None:
    """Verify LazyFrame.sink_iceberg is used for lazy writes."""
    feature_cls = test_features["UpstreamFeatureA"]

    metadata_lazy = pl.LazyFrame(
        {
            "sample_uid": [1, 2, 3],
            "metaxy_provenance_by_field": [
                {"frames": "h1", "audio": "h1"},
                {"frames": "h2", "audio": "h2"},
                {"frames": "h3", "audio": "h3"},
            ],
        }
    )

    with iceberg_store.open("w") as store:
        with patch.object(pl.LazyFrame, "sink_iceberg", autospec=True) as mock_sink:
            mock_sink.side_effect = lambda self_lf, *args, **kwargs: None
            store.write(feature_cls, metadata_lazy)
        mock_sink.assert_called_once()
        args, kwargs = mock_sink.call_args
        assert len(args) == 2
        assert kwargs == {"mode": "append"}
