"""Apache Iceberg-specific tests.

Most Iceberg store functionality is tested via parametrized tests in StoreCases.
This module tests Iceberg-specific features:
- Table identifier generation
- Schema evolution
- Custom namespace
- LazyFrame sink_iceberg write path
"""

from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest
from packaging.version import Version

from metaxy.ext.metadata_stores.iceberg import IcebergMetadataStore
from metaxy.models.types import FeatureKey


def test_iceberg_table_identifier(tmp_path) -> None:
    """Verify _table_identifier maps feature keys to (namespace, table_name) tuples."""
    store = IcebergMetadataStore(warehouse=tmp_path / "iceberg")

    assert store._table_identifier(FeatureKey("feature")) == ("metaxy", "feature")
    assert store._table_identifier(FeatureKey("a/b/c")) == ("metaxy", "a__b__c")
    assert store._table_identifier(FeatureKey("my_feature/v1")) == ("metaxy", "my_feature__v1")


def test_iceberg_schema_evolution(tmp_path, test_features) -> None:
    """Verify schema evolution when writing with new columns."""
    store_path = tmp_path / "iceberg_store"
    feature_cls = test_features["UpstreamFeatureA"]

    with IcebergMetadataStore(warehouse=store_path).open("w") as store:
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


def test_iceberg_custom_namespace(tmp_path, test_features) -> None:
    """Verify custom namespace is used for table identifiers."""
    store_path = tmp_path / "iceberg_store"
    feature_cls = test_features["UpstreamFeatureA"]

    with IcebergMetadataStore(warehouse=store_path, namespace="custom_ns").open("w") as store:
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
def test_iceberg_sink_lazyframe(tmp_path, test_features) -> None:
    """Verify LazyFrame.sink_iceberg is used for lazy writes."""
    store_path = tmp_path / "iceberg_store"
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

    with IcebergMetadataStore(warehouse=store_path).open("w") as store:
        with patch.object(pl.LazyFrame, "sink_iceberg", autospec=True) as mock_sink:
            mock_sink.side_effect = lambda self_lf, *args, **kwargs: None
            store.write(feature_cls, metadata_lazy)
        mock_sink.assert_called_once()
        args, kwargs = mock_sink.call_args
        assert len(args) == 2
        assert kwargs == {"mode": "append"}
