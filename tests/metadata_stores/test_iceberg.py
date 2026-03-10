"""Apache Iceberg-specific tests.

Most Iceberg store functionality is tested via parametrized tests in StoreCases.
This module tests Iceberg-specific features:
- Local warehouse path handling
- Table identifier generation (flat vs nested layout)
- Schema evolution
"""

from __future__ import annotations

import polars as pl

from metaxy.ext.metadata_stores.iceberg import IcebergMetadataStore
from metaxy.models.types import FeatureKey


def test_iceberg_local_absolute_path(tmp_path, test_features) -> None:
    """Verify IcebergMetadataStore works with local absolute paths."""
    store_path = tmp_path / "iceberg_store"
    feature_cls = test_features["UpstreamFeatureA"]

    with IcebergMetadataStore(warehouse=store_path).open("w") as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write(feature_cls, metadata)

        result = store.read(feature_cls)
        assert result is not None
        assert result.collect().to_native().height == 1


def test_iceberg_table_identifier_flat(tmp_path) -> None:
    """Verify _table_identifier generates correct identifiers in flat layout."""
    store = IcebergMetadataStore(warehouse=tmp_path / "iceberg", layout="flat")

    assert store._table_identifier(FeatureKey("feature")) == ("metaxy", "feature")
    assert store._table_identifier(FeatureKey("a/b/c")) == ("metaxy", "a__b__c")
    assert store._table_identifier(FeatureKey("my_feature/v1")) == ("metaxy", "my_feature__v1")


def test_iceberg_table_identifier_nested(tmp_path) -> None:
    """Verify _table_identifier generates correct identifiers in nested layout."""
    store = IcebergMetadataStore(warehouse=tmp_path / "iceberg", layout="nested")

    assert store._table_identifier(FeatureKey("feature")) == ("metaxy", "feature")
    assert store._table_identifier(FeatureKey("a/b/c")) == ("metaxy", "a", "b", "c")
    assert store._table_identifier(FeatureKey("my_feature/v1")) == ("metaxy", "my_feature", "v1")


def test_iceberg_schema_evolution(tmp_path, test_features) -> None:
    """Verify schema evolution when writing with new columns."""
    store_path = tmp_path / "iceberg_store"
    feature_cls = test_features["UpstreamFeatureA"]

    with IcebergMetadataStore(warehouse=store_path).open("w") as store:
        # Write initial data
        store.write(
            feature_cls,
            pl.DataFrame(
                {
                    "sample_uid": [1],
                    "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
                }
            ),
        )

        # Write with an extra column
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
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write(feature_cls, metadata)

        assert store.has_feature(feature_cls, check_fallback=False)

        result = store.read(feature_cls)
        assert result is not None
        assert result.collect().to_native().height == 1


def test_iceberg_drop_feature(tmp_path, test_features) -> None:
    """Verify dropping a feature removes the table."""
    store_path = tmp_path / "iceberg_store"
    feature_cls = test_features["UpstreamFeatureA"]

    with IcebergMetadataStore(warehouse=store_path).open("w") as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write(feature_cls, metadata)
        assert store.has_feature(feature_cls, check_fallback=False)

        store.drop_feature_metadata(feature_cls)
        assert not store.has_feature(feature_cls, check_fallback=False)


def test_iceberg_multiple_appends(tmp_path, test_features) -> None:
    """Verify multiple appends accumulate data."""
    store_path = tmp_path / "iceberg_store"
    feature_cls = test_features["UpstreamFeatureA"]

    with IcebergMetadataStore(warehouse=store_path).open("w") as store:
        for i in range(3):
            store.write(
                feature_cls,
                pl.DataFrame(
                    {
                        "sample_uid": [i],
                        "metaxy_provenance_by_field": [{"frames": f"h{i}", "audio": f"h{i}"}],
                    }
                ),
            )

        result = store.read(feature_cls)
        assert result is not None
        assert result.collect().to_native().height == 3
