"""Tests for SQLAlchemy integration helpers.

This module tests the SQLAlchemy integration functionality including:
1. System table metadata retrieval
2. Project-based filtering of user feature tables
3. MetadataStore sqlalchemy_url retrieval
"""

import pytest
from sqlmodel import Field

from metaxy import FeatureKey, FeatureSpec, FieldKey, FieldSpec
from metaxy.config import MetaxyConfig
from metaxy.ext.sqlalchemy import (
    filter_feature_sqla_metadata,
    get_system_slqa_metadata,
)
from metaxy.ext.sqlmodel import BaseSQLModelFeature
from metaxy.metadata_store.ibis import IbisMetadataStore

# Test fixtures and helper classes


@pytest.fixture(scope="function", autouse=False)
def _clear_sqlmodel_metadata():
    """Clear SQLModel metadata before tests that define Features."""
    from sqlmodel import SQLModel

    # Clear all tables from the metadata before the test
    SQLModel.metadata.clear()

    yield

    # Clear again after the test
    SQLModel.metadata.clear()


@pytest.fixture
def config_project_a(_clear_sqlmodel_metadata):
    """Config for project A with both project_a and project_b features defined."""
    config = MetaxyConfig(
        project="project_a",
        store="test_store",
        stores={  # pyright: ignore[reportArgumentType]
            "test_store": {
                "type": "metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                "config": {"database": ":memory:"},
            }
        },
    )
    with config.use():
        # Define ProjectAFeature within the config context
        class ProjectAFeature(
            BaseSQLModelFeature,
            table=True,
            spec=FeatureSpec(
                key=FeatureKey(["project_a", "feature"]),
                id_columns=["sample_uid"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            """Feature for project A."""

            sample_uid: str = Field(primary_key=True)
            value: str

        # Also define ProjectBFeature (for explicit_project test)
        # Need to temporarily switch project context
        with MetaxyConfig(
            project="project_b",
        ).use():

            class ProjectBFeature(
                BaseSQLModelFeature,
                table=True,
                spec=FeatureSpec(
                    key=FeatureKey(["project_b", "feature"]),
                    id_columns=["sample_uid"],
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                """Feature for project B."""

                sample_uid: str = Field(primary_key=True)
                value: str

        yield config


@pytest.fixture
def config_project_b(_clear_sqlmodel_metadata):
    """Config for project B."""
    config = MetaxyConfig(
        project="project_b",
        store="test_store",
        stores={  # pyright: ignore[reportArgumentType]
            "test_store": {
                "type": "metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                "config": {"database": ":memory:"},
            }
        },
    )
    with config.use():
        # Define ProjectBFeature within the config context
        class ProjectBFeature(
            BaseSQLModelFeature,
            table=True,
            spec=FeatureSpec(
                key=FeatureKey(["project_b", "feature"]),
                id_columns=["sample_uid"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            """Feature for project B."""

            sample_uid: str = Field(primary_key=True)
            value: str

        yield config


@pytest.fixture
def config_no_filter(_clear_sqlmodel_metadata):
    """Config with filtering disabled."""
    config = MetaxyConfig(
        project="project_a",
        store="test_store",
        stores={  # pyright: ignore[reportArgumentType]
            "test_store": {
                "type": "metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                "config": {"database": ":memory:"},
            }
        },
    )
    with config.use():
        # Define both features for the no-filter test
        class ProjectAFeature(
            BaseSQLModelFeature,
            table=True,
            spec=FeatureSpec(
                key=FeatureKey(["project_a", "feature"]),
                id_columns=["sample_uid"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            """Feature for project A."""

            sample_uid: str = Field(primary_key=True)
            value: str

        class ProjectBFeature(
            BaseSQLModelFeature,
            table=True,
            spec=FeatureSpec(
                key=FeatureKey(["project_b", "feature"]),
                id_columns=["sample_uid"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            """Feature for project B."""

            sample_uid: str = Field(primary_key=True)
            value: str

        yield config


# Tests


def test_get_metaxy_system_metadata():
    """Test retrieving system table metadata."""
    config = MetaxyConfig(
        project="test",
        store="test_store",
        stores={  # pyright: ignore[reportArgumentType]
            "test_store": {
                "type": "metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                "config": {"database": ":memory:"},
            }
        },
    )

    with config.use():
        store = config.get_store(expected_type=IbisMetadataStore)
        url, metadata = get_system_slqa_metadata(store)

        # Should contain system tables
        assert "metaxy_system__feature_versions" in metadata.tables
        assert "metaxy_system__events" in metadata.tables


def test_feature_versions_table_columns_match_polars_schema():
    """Regression test: SQLAlchemy feature_versions table must have all columns from Polars schema.

    This test ensures that the SQLAlchemy table definition in create_system_tables()
    stays in sync with FEATURE_VERSIONS_SCHEMA defined in models.py.
    Missing columns (like feature_spec, feature_class_path) would cause runtime errors
    when writing to the database.
    """
    from metaxy.metadata_store.system.models import FEATURE_VERSIONS_SCHEMA

    config = MetaxyConfig(
        project="test",
        store="test_store",
        stores={  # pyright: ignore[reportArgumentType]
            "test_store": {
                "type": "metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                "config": {"database": ":memory:"},
            }
        },
    )

    with config.use():
        store = config.get_store(expected_type=IbisMetadataStore)
        _, metadata = get_system_slqa_metadata(store)

        feature_versions_table = metadata.tables["metaxy_system__feature_versions"]
        sqla_column_names = {col.name for col in feature_versions_table.columns}
        polars_column_names = set(FEATURE_VERSIONS_SCHEMA.keys())

        # All Polars schema columns must exist in SQLAlchemy table
        missing_columns = polars_column_names - sqla_column_names
        assert not missing_columns, (
            f"SQLAlchemy feature_versions table is missing columns: {missing_columns}. "
            f"Update create_system_tables() in plugin.py to include these columns."
        )


def test_get_features_sqlalchemy_metadata_filtered_by_project(config_project_a):
    """Test that user features are filtered by project."""
    from sqlmodel import SQLModel

    store = config_project_a.get_store()
    url, metadata = filter_feature_sqla_metadata(store, SQLModel.metadata)

    # Should only contain project_a features
    assert "project_a__feature" in metadata.tables
    assert "project_b__feature" not in metadata.tables


def test_get_features_sqlalchemy_metadata_different_project(config_project_b):
    """Test filtering for a different project."""
    from sqlmodel import SQLModel

    store = config_project_b.get_store()
    url, metadata = filter_feature_sqla_metadata(store, SQLModel.metadata)

    # Should only contain project_b features
    assert "project_b__feature" in metadata.tables
    assert "project_a__feature" not in metadata.tables


def test_get_features_sqlalchemy_metadata_explicit_project(config_project_a):
    """Test explicit project parameter."""
    from sqlmodel import SQLModel

    # Get metadata for project_b even though current project is project_a
    store = config_project_a.get_store()
    url, metadata = filter_feature_sqla_metadata(
        store, SQLModel.metadata, project="project_b"
    )

    assert "project_b__feature" in metadata.tables
    assert "project_a__feature" not in metadata.tables


def test_get_features_sqlalchemy_metadata_no_filter(config_no_filter):
    """Test that filtering can be disabled."""
    from sqlmodel import SQLModel

    store = config_no_filter.get_store()
    url, metadata = filter_feature_sqla_metadata(
        store, SQLModel.metadata, filter_by_project=False
    )

    # Should contain all features when filtering is disabled
    assert "project_a__feature" in metadata.tables
    assert "project_b__feature" in metadata.tables


def test_get_store_sqlalchemy_url_default_store():
    """Test getting sqlalchemy_url from default store."""
    config = MetaxyConfig(
        project="test",
        store="test_store",
        stores={  # pyright: ignore[reportArgumentType]
            "test_store": {
                "type": "metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                "config": {"database": "test.db"},
            }
        },
    )

    with config.use():
        store = config.get_store(expected_type=IbisMetadataStore)
        url, metadata = get_system_slqa_metadata(store)
        assert url == "duckdb:///test.db"


def test_get_store_sqlalchemy_url_named_store():
    """Test getting sqlalchemy_url from named store."""
    config = MetaxyConfig(
        project="test",
        stores={  # pyright: ignore[reportArgumentType]
            "store1": {
                "type": "metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                "config": {"database": "store1.db"},
            },
            "store2": {
                "type": "metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                "config": {"database": "store2.db"},
            },
        },
    )

    with config.use():
        store1 = config.get_store("store1", expected_type=IbisMetadataStore)
        store2 = config.get_store("store2", expected_type=IbisMetadataStore)
        url1, metadata1 = get_system_slqa_metadata(store1)
        url2, metadata2 = get_system_slqa_metadata(store2)
        assert url1 == "duckdb:///store1.db"
        assert url2 == "duckdb:///store2.db"


@pytest.mark.skip(
    reason="Function now requires IbisMetadataStore - non-Ibis stores are not supported"
)
def test_get_store_sqlalchemy_url_error_no_property():
    """Test error when store doesn't support sqlalchemy_url."""
    config = MetaxyConfig(
        project="test",
        stores={  # pyright: ignore[reportArgumentType]
            "memory_store": {
                "type": "metaxy.metadata_store.memory.InMemoryMetadataStore",
                "config": {},
            }
        },
    )

    with config.use():
        store = config.get_store("memory_store", expected_type=IbisMetadataStore)  # type: ignore[arg-type]
        with pytest.raises(
            AttributeError, match="does not have a `sqlalchemy_url` property"
        ):
            get_system_slqa_metadata(store)


def test_get_store_sqlalchemy_url_error_no_url():
    """Test error when store has no URL available."""
    # DuckDBMetadataStore always has sqlalchemy_url, so we can't test ValueError with it
    # This test would only be relevant for a custom store that has sqlalchemy_url property
    # but returns None/empty. For now, skip this edge case as it's not realistic.
    pytest.skip("DuckDBMetadataStore always provides sqlalchemy_url")


class TestProtocolParameter:
    """Tests for the protocol parameter in SQLAlchemy URL functions."""

    def test_get_system_slqa_metadata_protocol_override(self):
        """Test that protocol parameter replaces the URL protocol."""
        config = MetaxyConfig(
            project="test",
            store="test_store",
            stores={  # pyright: ignore[reportArgumentType]
                "test_store": {
                    "type": "metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                    "config": {"database": "test.db"},
                }
            },
        )

        with config.use():
            store = config.get_store(expected_type=IbisMetadataStore)

            # Without protocol override
            url_default, _ = get_system_slqa_metadata(store)
            assert url_default == "duckdb:///test.db"

            # With protocol override
            url_override, _ = get_system_slqa_metadata(
                store, protocol="clickhouse+native"
            )
            assert url_override == "clickhouse+native:///test.db"

    def test_get_system_slqa_metadata_protocol_none_preserves_original(self):
        """Test that protocol=None preserves the original URL."""
        config = MetaxyConfig(
            project="test",
            store="test_store",
            stores={  # pyright: ignore[reportArgumentType]
                "test_store": {
                    "type": "metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                    "config": {"database": "mydb.db"},
                }
            },
        )

        with config.use():
            store = config.get_store(expected_type=IbisMetadataStore)
            url, _ = get_system_slqa_metadata(store, protocol=None)
            assert url == "duckdb:///mydb.db"

    def test_filter_feature_sqla_metadata_protocol_override(self, config_project_a):
        """Test that protocol parameter works in filter_feature_sqla_metadata."""
        from sqlmodel import SQLModel

        store = config_project_a.get_store()

        # Without protocol override - should be duckdb
        url_default, _ = filter_feature_sqla_metadata(store, SQLModel.metadata)
        assert url_default.startswith("duckdb://")

        # With protocol override
        url_override, metadata = filter_feature_sqla_metadata(
            store, SQLModel.metadata, protocol="postgresql+psycopg2"
        )
        assert url_override.startswith("postgresql+psycopg2://")
        # Metadata should still be filtered correctly
        assert "project_a__feature" in metadata.tables

    def test_protocol_override_preserves_url_components(self):
        """Test that protocol override preserves host, port, database, etc."""
        from unittest.mock import MagicMock

        from metaxy.ext.sqlalchemy.plugin import _get_store_sqlalchemy_url

        # Create a mock store with a complex URL
        mock_store = MagicMock(spec=IbisMetadataStore)
        mock_store.sqlalchemy_url = (
            "clickhouse://user:pass@localhost:9000/mydb?secure=true"
        )

        # Test protocol replacement preserves everything after ://
        url = _get_store_sqlalchemy_url(mock_store, protocol="clickhouse+native")
        assert url == "clickhouse+native://user:pass@localhost:9000/mydb?secure=true"

    def test_protocol_override_various_protocols(self):
        """Test protocol override with various protocol names."""
        from unittest.mock import MagicMock

        from metaxy.ext.sqlalchemy.plugin import _get_store_sqlalchemy_url

        mock_store = MagicMock(spec=IbisMetadataStore)
        mock_store.sqlalchemy_url = "original://host:1234/db"

        test_cases = [
            ("clickhouse+native", "clickhouse+native://host:1234/db"),
            ("clickhouse+http", "clickhouse+http://host:1234/db"),
            ("postgresql", "postgresql://host:1234/db"),
            ("mysql+pymysql", "mysql+pymysql://host:1234/db"),
        ]

        for protocol, expected in test_cases:
            result = _get_store_sqlalchemy_url(mock_store, protocol=protocol)
            assert result == expected, f"Failed for protocol={protocol}"

    def test_protocol_override_empty_store_url_raises(self):
        """Test that empty sqlalchemy_url raises ValueError regardless of protocol."""
        from unittest.mock import MagicMock

        from metaxy.ext.sqlalchemy.plugin import _get_store_sqlalchemy_url

        mock_store = MagicMock(spec=IbisMetadataStore)
        mock_store.sqlalchemy_url = ""

        with pytest.raises(ValueError, match="empty"):
            _get_store_sqlalchemy_url(mock_store, protocol="clickhouse+native")

    def test_port_override(self):
        """Test that port parameter replaces the URL port."""
        from unittest.mock import MagicMock

        from metaxy.ext.sqlalchemy.plugin import _get_store_sqlalchemy_url

        mock_store = MagicMock(spec=IbisMetadataStore)
        mock_store.sqlalchemy_url = "clickhouse://localhost:8443/default"

        # Test port replacement
        url = _get_store_sqlalchemy_url(mock_store, port=9000)
        assert url == "clickhouse://localhost:9000/default"

    def test_protocol_and_port_override_together(self):
        """Test that both protocol and port can be overridden together."""
        from unittest.mock import MagicMock

        from metaxy.ext.sqlalchemy.plugin import _get_store_sqlalchemy_url

        mock_store = MagicMock(spec=IbisMetadataStore)
        mock_store.sqlalchemy_url = "clickhouse://user:pass@localhost:8443/mydb"

        # Test both protocol and port replacement
        url = _get_store_sqlalchemy_url(
            mock_store, protocol="clickhouse+native", port=9000
        )
        assert url == "clickhouse+native://user:pass@localhost:9000/mydb"

    def test_port_override_preserves_other_components(self):
        """Test that port override preserves all other URL components."""
        from unittest.mock import MagicMock

        from metaxy.ext.sqlalchemy.plugin import _get_store_sqlalchemy_url

        mock_store = MagicMock(spec=IbisMetadataStore)
        mock_store.sqlalchemy_url = (
            "clickhouse://user:pass@localhost:8443/mydb?secure=true"
        )

        url = _get_store_sqlalchemy_url(mock_store, port=9000)
        assert url == "clickhouse://user:pass@localhost:9000/mydb?secure=true"
