"""Tests for SQLAlchemy integration helpers.

This module tests the SQLAlchemy integration functionality including:
1. System table metadata retrieval
2. Project-based filtering of user feature tables
3. MetadataStore sqlalchemy_url retrieval
"""

from unittest.mock import MagicMock

import pytest
from sqlalchemy import Column, MetaData, String, Table

from metaxy import BaseFeature, FeatureKey, FeatureSpec, FieldKey, FieldSpec
from metaxy.config import MetaxyConfig
from metaxy.ext.sqlalchemy import (
    filter_feature_sqla_metadata,
    get_system_slqa_metadata,
)
from metaxy.ext.sqlalchemy.plugin import _get_store_sqlalchemy_url
from metaxy.metadata_store.ibis import IbisMetadataStore

# Test fixtures


@pytest.fixture
def config_with_features():
    """Config with two features from different projects, using plain SQLAlchemy MetaData."""
    config = MetaxyConfig(
        project="project_a",
        store="test_store",
        stores={  # ty: ignore[invalid-argument-type]
            "test_store": {
                "type": "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore",
                "config": {"database": ":memory:"},
            }
        },
    )
    with config.use():

        class FeatureA(
            BaseFeature,
            spec=FeatureSpec(
                key=FeatureKey(["project_a", "feature"]),
                id_columns=["sample_uid"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            __metaxy_project__ = "project_a"
            sample_uid: str
            value: str

        class FeatureB(
            BaseFeature,
            spec=FeatureSpec(
                key=FeatureKey(["project_b", "feature"]),
                id_columns=["sample_uid"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            __metaxy_project__ = "project_b"
            sample_uid: str
            value: str

        yield config


@pytest.fixture
def sqla_metadata(config_with_features: MetaxyConfig) -> MetaData:
    """Plain SQLAlchemy MetaData with tables matching the feature table names."""
    store = config_with_features.get_store()
    metadata = MetaData()
    for feature_key in [FeatureKey(["project_a", "feature"]), FeatureKey(["project_b", "feature"])]:
        Table(
            store.get_table_name(feature_key),
            metadata,
            Column("sample_uid", String, primary_key=True),
            Column("value", String),
        )
    return metadata


# Tests


def test_get_metaxy_system_metadata():
    """Test retrieving system table metadata."""
    config = MetaxyConfig(
        project="test",
        store="test_store",
        stores={  # ty: ignore[invalid-argument-type]
            "test_store": {
                "type": "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore",
                "config": {"database": ":memory:"},
            }
        },
    )

    with config.use():
        store = config.get_store()
        url, metadata = get_system_slqa_metadata(store)

        assert "metaxy_system__feature_versions" in metadata.tables
        assert "metaxy_system__events" in metadata.tables


def test_feature_versions_table_columns_match_polars_schema():
    """Regression test: SQLAlchemy feature_versions table must have all columns from Polars schema."""
    from metaxy.metadata_store.system.models import FEATURE_VERSIONS_SCHEMA

    config = MetaxyConfig(
        project="test",
        store="test_store",
        stores={  # ty: ignore[invalid-argument-type]
            "test_store": {
                "type": "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore",
                "config": {"database": ":memory:"},
            }
        },
    )

    with config.use():
        store = config.get_store()
        _, metadata = get_system_slqa_metadata(store)

        feature_versions_table = metadata.tables["metaxy_system__feature_versions"]
        sqla_column_names = {col.name for col in feature_versions_table.columns}
        polars_column_names = set(FEATURE_VERSIONS_SCHEMA.keys())

        missing_columns = polars_column_names - sqla_column_names
        assert not missing_columns, (
            f"SQLAlchemy feature_versions table is missing columns: {missing_columns}. "
            f"Update create_system_tables() in plugin.py to include these columns."
        )


def test_filter_feature_sqla_metadata_filtered_by_project(
    config_with_features: MetaxyConfig,
    sqla_metadata: MetaData,
):
    """Test that user features are filtered by project."""
    store = config_with_features.get_store()
    _, metadata = filter_feature_sqla_metadata(store, sqla_metadata)

    assert "project_a__feature" in metadata.tables
    assert "project_b__feature" not in metadata.tables


def test_filter_feature_sqla_metadata_explicit_project(
    config_with_features: MetaxyConfig,
    sqla_metadata: MetaData,
):
    """Test explicit project parameter."""
    store = config_with_features.get_store()
    _, metadata = filter_feature_sqla_metadata(store, sqla_metadata, project="project_b")

    assert "project_b__feature" in metadata.tables
    assert "project_a__feature" not in metadata.tables


def test_filter_feature_sqla_metadata_no_filter(
    config_with_features: MetaxyConfig,
    sqla_metadata: MetaData,
):
    """Test that filtering can be disabled."""
    store = config_with_features.get_store()
    _, metadata = filter_feature_sqla_metadata(store, sqla_metadata, filter_by_project=False)

    assert "project_a__feature" in metadata.tables
    assert "project_b__feature" in metadata.tables


def test_get_store_sqlalchemy_url_default_store():
    """Test getting sqlalchemy_url from default store."""
    config = MetaxyConfig(
        project="test",
        store="test_store",
        stores={  # ty: ignore[invalid-argument-type]
            "test_store": {
                "type": "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore",
                "config": {"database": "test.db"},
            }
        },
    )

    with config.use():
        store = config.get_store()
        url, _ = get_system_slqa_metadata(store)
        assert url == "duckdb:///test.db"


def test_get_store_sqlalchemy_url_named_store():
    """Test getting sqlalchemy_url from named store."""
    config = MetaxyConfig(
        project="test",
        stores={  # ty: ignore[invalid-argument-type]
            "store1": {
                "type": "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore",
                "config": {"database": "store1.db"},
            },
            "store2": {
                "type": "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore",
                "config": {"database": "store2.db"},
            },
        },
    )

    with config.use():
        store1 = config.get_store("store1")
        store2 = config.get_store("store2")
        url1, _ = get_system_slqa_metadata(store1)
        url2, _ = get_system_slqa_metadata(store2)
        assert url1 == "duckdb:///store1.db"
        assert url2 == "duckdb:///store2.db"


class TestProtocolParameter:
    """Tests for the protocol parameter in SQLAlchemy URL functions."""

    def test_get_system_slqa_metadata_protocol_override(self):
        """Test that protocol parameter replaces the URL protocol."""
        config = MetaxyConfig(
            project="test",
            store="test_store",
            stores={  # ty: ignore[invalid-argument-type]
                "test_store": {
                    "type": "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore",
                    "config": {"database": "test.db"},
                }
            },
        )

        with config.use():
            store = config.get_store()

            url_default, _ = get_system_slqa_metadata(store)
            assert url_default == "duckdb:///test.db"

            url_override, _ = get_system_slqa_metadata(store, protocol="clickhouse+native")
            assert url_override == "clickhouse+native:///test.db"

    def test_get_system_slqa_metadata_protocol_none_preserves_original(self):
        """Test that protocol=None preserves the original URL."""
        config = MetaxyConfig(
            project="test",
            store="test_store",
            stores={  # ty: ignore[invalid-argument-type]
                "test_store": {
                    "type": "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore",
                    "config": {"database": "mydb.db"},
                }
            },
        )

        with config.use():
            store = config.get_store()
            url, _ = get_system_slqa_metadata(store, protocol=None)
            assert url == "duckdb:///mydb.db"

    def test_filter_feature_sqla_metadata_protocol_override(
        self,
        config_with_features: MetaxyConfig,
        sqla_metadata: MetaData,
    ):
        """Test that protocol parameter works in filter_feature_sqla_metadata."""
        store = config_with_features.get_store()

        url_default, _ = filter_feature_sqla_metadata(store, sqla_metadata)
        assert url_default.startswith("duckdb://")

        url_override, metadata = filter_feature_sqla_metadata(store, sqla_metadata, protocol="postgresql+psycopg2")
        assert url_override.startswith("postgresql+psycopg2://")
        assert "project_a__feature" in metadata.tables

    def test_protocol_override_preserves_url_components(self):
        """Test that protocol override preserves host, port, database, etc."""
        mock_store = MagicMock(spec=IbisMetadataStore)
        mock_store.sqlalchemy_url = "clickhouse://user:pass@localhost:9000/mydb?secure=true"

        url = _get_store_sqlalchemy_url(mock_store, protocol="clickhouse+native")
        assert url == "clickhouse+native://user:pass@localhost:9000/mydb?secure=true"

    def test_protocol_override_various_protocols(self):
        """Test protocol override with various protocol names."""
        mock_store = MagicMock(spec=IbisMetadataStore)
        mock_store.sqlalchemy_url = "original://host:1234/db"

        test_cases = [
            ("clickhouse+native", "clickhouse+native://host:1234/db"),
            ("clickhouse+http", "clickhouse+http://host:1234/db"),
            ("postgresql", "postgresql://host:1234/db"),
            ("mysql+pymysql", "mysql+pymysql://host:1234/db"),
        ]

        for protocol, expected in test_cases:
            assert _get_store_sqlalchemy_url(mock_store, protocol=protocol) == expected

    def test_protocol_override_empty_store_url_raises(self):
        """Test that empty sqlalchemy_url raises ValueError regardless of protocol."""
        mock_store = MagicMock(spec=IbisMetadataStore)
        mock_store.sqlalchemy_url = ""

        with pytest.raises(ValueError, match="empty"):
            _get_store_sqlalchemy_url(mock_store, protocol="clickhouse+native")

    def test_port_override(self):
        """Test that port parameter replaces the URL port."""
        mock_store = MagicMock(spec=IbisMetadataStore)
        mock_store.sqlalchemy_url = "clickhouse://localhost:8443/default"

        assert _get_store_sqlalchemy_url(mock_store, port=9000) == "clickhouse://localhost:9000/default"

    def test_protocol_and_port_override_together(self):
        """Test that both protocol and port can be overridden together."""
        mock_store = MagicMock(spec=IbisMetadataStore)
        mock_store.sqlalchemy_url = "clickhouse://user:pass@localhost:8443/mydb"

        url = _get_store_sqlalchemy_url(mock_store, protocol="clickhouse+native", port=9000)
        assert url == "clickhouse+native://user:pass@localhost:9000/mydb"

    def test_port_override_preserves_other_components(self):
        """Test that port override preserves all other URL components."""
        mock_store = MagicMock(spec=IbisMetadataStore)
        mock_store.sqlalchemy_url = "clickhouse://user:pass@localhost:8443/mydb?secure=true"

        assert (
            _get_store_sqlalchemy_url(mock_store, port=9000) == "clickhouse://user:pass@localhost:9000/mydb?secure=true"
        )
