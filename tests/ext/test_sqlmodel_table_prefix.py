"""Tests for SQLModel table_prefix handling in filter_feature_sqlmodel_metadata."""

import pytest
from sqlmodel import Field, SQLModel

from metaxy import FeatureKey, FeatureSpec, FieldKey, FieldSpec
from metaxy.config import MetaxyConfig, StoreConfig
from metaxy.ext.sqlmodel import (
    BaseSQLModelFeature,
    SQLModelPluginConfig,
    filter_feature_sqlmodel_metadata,
)
from metaxy.metadata_store.ibis import IbisMetadataStore


@pytest.fixture(scope="function", autouse=True)
def _clear_sqlmodel_metadata():
    """Clear SQLModel metadata before each test."""
    SQLModel.metadata.clear()
    yield
    SQLModel.metadata.clear()


def test_filter_feature_sqlmodel_metadata_applies_table_prefix():
    """Test that filter_feature_sqlmodel_metadata applies store's table_prefix to table names."""

    # Create config with a store that has table_prefix
    config = MetaxyConfig(
        project="test_project",
        stores={
            "prefixed_store": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={
                    "database": ":memory:",
                    "table_prefix": "myprefix_",
                },
            )
        },
        store="prefixed_store",
        ext={"sqlmodel": SQLModelPluginConfig(enable=True)},
    )

    with config.use():
        # Define a SQLModel feature AFTER setting config
        class TestFeature(
            BaseSQLModelFeature,
            table=True,
            spec=FeatureSpec(
                key=FeatureKey(["test", "feature"]),
                id_columns=["id"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            project = "test_project"  # Must match config project
            id: str = Field(primary_key=True)
            value: str

        store = config.get_store(expected_type=IbisMetadataStore)

        # Filter SQLModel metadata
        url, metadata = filter_feature_sqlmodel_metadata(store, SQLModel.metadata)

        # Check that table name includes prefix
        # Original __tablename__ is "test__feature" (from FeatureKey)
        # With prefix "myprefix_", it should be "myprefix_test__feature"
        assert "myprefix_test__feature" in metadata.tables
        assert "test__feature" not in metadata.tables


def test_filter_feature_sqlmodel_metadata_no_prefix():
    """Test that filter_feature_sqlmodel_metadata works with empty table_prefix."""

    # Create config with a store that has no table_prefix
    config = MetaxyConfig(
        project="test_project",
        stores={
            "no_prefix_store": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={
                    "database": ":memory:",
                },
            )
        },
        store="no_prefix_store",
        ext={"sqlmodel": SQLModelPluginConfig(enable=True)},
    )

    with config.use():

        class TestFeature2(
            BaseSQLModelFeature,
            table=True,
            spec=FeatureSpec(
                key=FeatureKey(["test", "feature2"]),
                id_columns=["id"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            project = "test_project"
            id: str = Field(primary_key=True)
            value: str

        store = config.get_store(expected_type=IbisMetadataStore)

        # Filter SQLModel metadata
        url, metadata = filter_feature_sqlmodel_metadata(store, SQLModel.metadata)

        # Check that table name has no prefix
        assert "test__feature2" in metadata.tables


def test_filter_feature_sqlmodel_metadata_different_prefixes():
    """Test that different stores with different prefixes produce different metadata."""

    # Create config with multiple stores with different prefixes
    config = MetaxyConfig(
        project="test_project",
        stores={
            "store_a": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={
                    "database": ":memory:",
                    "table_prefix": "a_",
                },
            ),
            "store_b": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={
                    "database": ":memory:",
                    "table_prefix": "b_",
                },
            ),
        },
        store="store_a",
        ext={"sqlmodel": SQLModelPluginConfig(enable=True)},
    )

    with config.use():

        class MultiStoreFeature(
            BaseSQLModelFeature,
            table=True,
            spec=FeatureSpec(
                key=FeatureKey(["multi", "store"]),
                id_columns=["id"],
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            project = "test_project"
            id: str = Field(primary_key=True)
            data: str

        # Get metadata for store_a
        store_a = config.get_store("store_a", expected_type=IbisMetadataStore)
        url_a, metadata_a = filter_feature_sqlmodel_metadata(store_a, SQLModel.metadata)

        # Get metadata for store_b
        store_b = config.get_store("store_b", expected_type=IbisMetadataStore)
        url_b, metadata_b = filter_feature_sqlmodel_metadata(store_b, SQLModel.metadata)

        # Check that each metadata has the correct prefix
        assert "a_multi__store" in metadata_a.tables
        assert "b_multi__store" not in metadata_a.tables

        assert "b_multi__store" in metadata_b.tables
        assert "a_multi__store" not in metadata_b.tables


def test_filter_feature_sqlmodel_metadata_with_project_filtering_and_prefix():
    """Test that project filtering and prefix transformation work together.

    Note: Both features will be auto-assigned project from config context,
    so we test filter_by_project=False to include all features.
    """

    # Create config with prefix
    config = MetaxyConfig(
        project="test_project",
        stores={
            "prefixed_store": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={
                    "database": ":memory:",
                    "table_prefix": "env_",
                },
            )
        },
        store="prefixed_store",
        ext={"sqlmodel": SQLModelPluginConfig(enable=True)},
    )

    with config.use():

        class Feature1(
            BaseSQLModelFeature,
            table=True,
            spec=FeatureSpec(
                key=FeatureKey(["feature1"]),
                id_columns=["id"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            # Will auto-detect project from config = "test_project"
            id: str = Field(primary_key=True)
            value: str

        class Feature2(
            BaseSQLModelFeature,
            table=True,
            spec=FeatureSpec(
                key=FeatureKey(["feature2"]),
                id_columns=["id"],
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            # Will auto-detect project from config = "test_project"
            id: str = Field(primary_key=True)
            data: str

        store = config.get_store(expected_type=IbisMetadataStore)

        # Filter with project filtering enabled (default) - should include both since they're in same project
        url, metadata = filter_feature_sqlmodel_metadata(store, SQLModel.metadata)

        # Both features should be included with prefix
        assert "env_feature1" in metadata.tables
        assert "env_feature2" in metadata.tables

        # No unprefixed versions
        assert "feature1" not in metadata.tables
        assert "feature2" not in metadata.tables


def test_custom_tablename_raises_error():
    """Test that defining custom __tablename__ raises an error.

    Custom __tablename__ is forbidden because it would be inconsistent with
    the metadata store's get_table_name() method.
    """

    # Create config
    config = MetaxyConfig(
        project="test_project",
        stores={
            "default_store": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": ":memory:"},
            )
        },
        store="default_store",
        ext={"sqlmodel": SQLModelPluginConfig(enable=True)},
    )

    with config.use():
        # Attempting to define custom __tablename__ should raise ValueError
        with pytest.raises(ValueError, match="Cannot define custom __tablename__"):

            class CustomTableFeature(
                BaseSQLModelFeature,
                table=True,
                spec=FeatureSpec(
                    key=FeatureKey(["my", "feature"]),
                    id_columns=["id"],
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                __tablename__: str = "my_custom_table"
                project = "test_project"
                id: str = Field(primary_key=True)
                value: str


class TestProtocolParameter:
    """Tests for the protocol parameter in filter_feature_sqlmodel_metadata."""

    def test_protocol_override_replaces_url_protocol(self):
        """Test that protocol parameter replaces the URL protocol."""
        config = MetaxyConfig(
            project="test_project",
            stores={
                "test_store": StoreConfig(
                    type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                    config={"database": "test.db"},
                )
            },
            store="test_store",
            ext={"sqlmodel": SQLModelPluginConfig(enable=True)},
        )

        with config.use():

            class ProtocolTestFeature(
                BaseSQLModelFeature,
                table=True,
                spec=FeatureSpec(
                    key=FeatureKey(["protocol", "test"]),
                    id_columns=["id"],
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                id: str = Field(primary_key=True)
                value: str

            store = config.get_store(expected_type=IbisMetadataStore)

            # Without protocol override
            url_default, _ = filter_feature_sqlmodel_metadata(store, SQLModel.metadata)
            assert url_default == "duckdb:///test.db"

            # With protocol override
            url_override, metadata = filter_feature_sqlmodel_metadata(
                store, SQLModel.metadata, protocol="clickhouse+native"
            )
            assert url_override == "clickhouse+native:///test.db"
            # Metadata should still be filtered correctly
            assert "protocol__test" in metadata.tables

    def test_protocol_none_preserves_original_url(self):
        """Test that protocol=None preserves the original URL."""
        config = MetaxyConfig(
            project="test_project",
            stores={
                "test_store": StoreConfig(
                    type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                    config={"database": "mydb.db"},
                )
            },
            store="test_store",
            ext={"sqlmodel": SQLModelPluginConfig(enable=True)},
        )

        with config.use():

            class ProtocolNoneFeature(
                BaseSQLModelFeature,
                table=True,
                spec=FeatureSpec(
                    key=FeatureKey(["protocol", "none"]),
                    id_columns=["id"],
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                id: str = Field(primary_key=True)
                value: str

            store = config.get_store(expected_type=IbisMetadataStore)
            url, _ = filter_feature_sqlmodel_metadata(
                store, SQLModel.metadata, protocol=None
            )
            assert url == "duckdb:///mydb.db"

    def test_protocol_override_with_table_prefix(self):
        """Test that protocol override works together with table_prefix."""
        config = MetaxyConfig(
            project="test_project",
            stores={
                "prefixed_store": StoreConfig(
                    type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                    config={
                        "database": "test.db",
                        "table_prefix": "myprefix_",
                    },
                )
            },
            store="prefixed_store",
            ext={"sqlmodel": SQLModelPluginConfig(enable=True)},
        )

        with config.use():

            class PrefixProtocolFeature(
                BaseSQLModelFeature,
                table=True,
                spec=FeatureSpec(
                    key=FeatureKey(["prefix", "protocol"]),
                    id_columns=["id"],
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                id: str = Field(primary_key=True)
                value: str

            store = config.get_store(expected_type=IbisMetadataStore)

            # With protocol override
            url, metadata = filter_feature_sqlmodel_metadata(
                store, SQLModel.metadata, protocol="postgresql+psycopg2"
            )

            # Protocol should be replaced
            assert url.startswith("postgresql+psycopg2://")

            # Table prefix should still be applied
            assert "myprefix_prefix__protocol" in metadata.tables
            assert "prefix__protocol" not in metadata.tables
