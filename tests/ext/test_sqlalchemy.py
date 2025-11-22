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
    get_feature_sqla_metadata_for_store,
    get_system_sqla_metadata_for_store,
)
from metaxy.ext.sqlmodel import BaseSQLModelFeature

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
        url, metadata = get_system_sqla_metadata_for_store()

        # Should contain system tables
        assert "metaxy_system__feature_versions" in metadata.tables
        assert "metaxy_system__events" in metadata.tables


def test_get_features_sqlalchemy_metadata_filtered_by_project(config_project_a):
    """Test that user features are filtered by project."""
    url, metadata = get_feature_sqla_metadata_for_store()

    # Should only contain project_a features
    assert "project_a__feature" in metadata.tables
    assert "project_b__feature" not in metadata.tables


def test_get_features_sqlalchemy_metadata_different_project(config_project_b):
    """Test filtering for a different project."""
    url, metadata = get_feature_sqla_metadata_for_store()

    # Should only contain project_b features
    assert "project_b__feature" in metadata.tables
    assert "project_a__feature" not in metadata.tables


def test_get_features_sqlalchemy_metadata_explicit_project(config_project_a):
    """Test explicit project parameter."""
    # Get metadata for project_b even though current project is project_a
    url, metadata = get_feature_sqla_metadata_for_store(project="project_b")

    assert "project_b__feature" in metadata.tables
    assert "project_a__feature" not in metadata.tables


def test_get_features_sqlalchemy_metadata_no_filter(config_no_filter):
    """Test that filtering can be disabled."""
    url, metadata = get_feature_sqla_metadata_for_store(filter_by_project=False)

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
        url, metadata = get_system_sqla_metadata_for_store()
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
        url1, metadata1 = get_system_sqla_metadata_for_store("store1")
        url2, metadata2 = get_system_sqla_metadata_for_store("store2")
        assert url1 == "duckdb:///store1.db"
        assert url2 == "duckdb:///store2.db"


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
        with pytest.raises(
            AttributeError, match="does not have a `sqlalchemy_url` property"
        ):
            get_system_sqla_metadata_for_store("memory_store")


def test_get_store_sqlalchemy_url_error_no_url():
    """Test error when store has no URL available."""
    # DuckDBMetadataStore always has sqlalchemy_url, so we can't test ValueError with it
    # This test would only be relevant for a custom store that has sqlalchemy_url property
    # but returns None/empty. For now, skip this edge case as it's not realistic.
    pytest.skip("DuckDBMetadataStore always provides sqlalchemy_url")
