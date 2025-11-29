"""Tests for error table metadata generation in SQLAlchemy/SQLModel filtering functions.

Error tables are system features that are automatically created for each user-defined feature.
They have the same id_columns as the parent feature but contain error data. These tests verify
that the filter_feature_sqla_metadata and filter_feature_sqlmodel_metadata functions properly
generate SQLAlchemy metadata for error tables.
"""

from __future__ import annotations

import pytest
from sqlmodel import Field, SQLModel

from metaxy import FeatureKey, FeatureSpec, FieldKey, FieldSpec
from metaxy.config import MetaxyConfig, StoreConfig
from metaxy.ext.sqlalchemy import filter_feature_sqla_metadata
from metaxy.ext.sqlmodel import (
    BaseSQLModelFeature,
    SQLModelPluginConfig,
    filter_feature_sqlmodel_metadata,
)
from metaxy.metadata_store.ibis import IbisMetadataStore
from metaxy.models.constants import (
    METAXY_CREATED_AT,
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_FEATURE_SPEC_VERSION,
    METAXY_FEATURE_VERSION,
    METAXY_MATERIALIZATION_ID,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
    METAXY_SNAPSHOT_VERSION,
)


@pytest.fixture(scope="function", autouse=True)
def _clear_sqlmodel_metadata():
    """Clear SQLModel metadata before each test."""
    SQLModel.metadata.clear()
    yield
    SQLModel.metadata.clear()


def test_error_table_metadata_included_for_sqlmodel():
    """Test that error tables get SQLAlchemy metadata when parent has SQLModel table."""
    config = MetaxyConfig(
        project="test_project",
        stores={
            "test_store": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": ":memory:"},
            )
        },
        store="test_store",
        ext={"sqlmodel": SQLModelPluginConfig(enable=True)},
    )

    with config.use():

        class ParentFeature(
            BaseSQLModelFeature,
            table=True,
            inject_primary_key=False,
            spec=FeatureSpec(
                key=FeatureKey(["parent", "feature"]),
                id_columns=["sample_uid"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            sample_uid: str = Field(primary_key=True)
            value: str

        store = config.get_store(expected_type=IbisMetadataStore)

        # Filter SQLModel metadata
        url, metadata = filter_feature_sqlmodel_metadata(store, SQLModel.metadata)

        # Check that parent feature table exists
        assert "parent__feature" in metadata.tables

        # Check that error table exists
        assert "parent__feature__errors" in metadata.tables


def test_error_table_metadata_included_for_sqlalchemy():
    """Test that error tables get SQLAlchemy metadata via filter_feature_sqla_metadata."""
    config = MetaxyConfig(
        project="test_project",
        stores={
            "test_store": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": ":memory:"},
            )
        },
        store="test_store",
        ext={"sqlmodel": SQLModelPluginConfig(enable=True)},
    )

    with config.use():

        class ParentFeatureSqla(
            BaseSQLModelFeature,
            table=True,
            inject_primary_key=False,
            spec=FeatureSpec(
                key=FeatureKey(["parent", "sqla"]),
                id_columns=["id"],
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            id: int = Field(primary_key=True)
            data: str

        store = config.get_store(expected_type=IbisMetadataStore)

        # Filter using SQLAlchemy function
        url, metadata = filter_feature_sqla_metadata(store, SQLModel.metadata)

        # Check that parent feature table exists
        assert "parent__sqla" in metadata.tables

        # Check that error table exists
        assert "parent__sqla__errors" in metadata.tables


def test_error_table_id_columns_match_parent_types():
    """Test that error table id_columns have the same types as parent's columns."""
    config = MetaxyConfig(
        project="test_project",
        stores={
            "test_store": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": ":memory:"},
            )
        },
        store="test_store",
        ext={"sqlmodel": SQLModelPluginConfig(enable=True)},
    )

    with config.use():

        class TypedParent(
            BaseSQLModelFeature,
            table=True,
            inject_primary_key=False,
            spec=FeatureSpec(
                key=FeatureKey(["typed", "parent"]),
                id_columns=["int_id", "str_id"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            int_id: int = Field(primary_key=True)
            str_id: str = Field(primary_key=True)
            value: float

        store = config.get_store(expected_type=IbisMetadataStore)
        url, metadata = filter_feature_sqlmodel_metadata(store, SQLModel.metadata)

        # Get the parent and error tables
        parent_table = metadata.tables["typed__parent"]
        error_table = metadata.tables["typed__parent__errors"]

        # Check that error table has the same id_column types as parent
        parent_int_col = parent_table.c.get("int_id")
        parent_str_col = parent_table.c.get("str_id")
        error_int_col = error_table.c.get("int_id")
        error_str_col = error_table.c.get("str_id")

        assert parent_int_col is not None
        assert parent_str_col is not None
        assert error_int_col is not None
        assert error_str_col is not None

        # Type comparison - they should be equivalent types
        assert type(error_int_col.type).__name__ == type(parent_int_col.type).__name__
        assert type(error_str_col.type).__name__ == type(parent_str_col.type).__name__


def test_error_table_has_default_field():
    """Test that error table has a 'default' field for error data."""
    config = MetaxyConfig(
        project="test_project",
        stores={
            "test_store": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": ":memory:"},
            )
        },
        store="test_store",
        ext={"sqlmodel": SQLModelPluginConfig(enable=True)},
    )

    with config.use():

        class FeatureWithErrors(
            BaseSQLModelFeature,
            table=True,
            inject_primary_key=False,
            spec=FeatureSpec(
                key=FeatureKey(["feature", "errors"]),
                id_columns=["id"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            id: str = Field(primary_key=True)
            value: str

        store = config.get_store(expected_type=IbisMetadataStore)
        url, metadata = filter_feature_sqlmodel_metadata(store, SQLModel.metadata)

        # Get the error table
        error_table = metadata.tables["feature__errors__errors"]

        # Check that error table has 'default' column
        default_col = error_table.c.get("default")
        assert default_col is not None
        # Default column should be JSON type
        assert "JSON" in type(default_col.type).__name__


def test_error_table_has_system_columns():
    """Test that error table has all required system columns."""
    config = MetaxyConfig(
        project="test_project",
        stores={
            "test_store": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": ":memory:"},
            )
        },
        store="test_store",
        ext={"sqlmodel": SQLModelPluginConfig(enable=True)},
    )

    with config.use():

        class SystemColsFeature(
            BaseSQLModelFeature,
            table=True,
            inject_primary_key=False,
            spec=FeatureSpec(
                key=FeatureKey(["syscols"]),
                id_columns=["id"],
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            id: str = Field(primary_key=True)
            data: str

        store = config.get_store(expected_type=IbisMetadataStore)
        url, metadata = filter_feature_sqlmodel_metadata(store, SQLModel.metadata)

        # Get the error table
        error_table = metadata.tables["syscols__errors"]

        # Check for all system columns
        expected_system_cols = [
            METAXY_PROVENANCE,
            METAXY_PROVENANCE_BY_FIELD,
            METAXY_FEATURE_VERSION,
            METAXY_FEATURE_SPEC_VERSION,
            METAXY_SNAPSHOT_VERSION,
            METAXY_DATA_VERSION,
            METAXY_DATA_VERSION_BY_FIELD,
            METAXY_CREATED_AT,
            METAXY_MATERIALIZATION_ID,
        ]

        for col_name in expected_system_cols:
            col = error_table.c.get(col_name)
            assert col is not None, f"Missing system column: {col_name}"


def test_error_table_with_table_prefix():
    """Test that error table names include the store's table_prefix."""
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

        class PrefixedFeature(
            BaseSQLModelFeature,
            table=True,
            inject_primary_key=False,
            spec=FeatureSpec(
                key=FeatureKey(["prefixed"]),
                id_columns=["id"],
                fields=[FieldSpec(key=FieldKey(["val"]), code_version="1")],
            ),
        ):
            id: str = Field(primary_key=True)
            val: str

        store = config.get_store(expected_type=IbisMetadataStore)
        url, metadata = filter_feature_sqlmodel_metadata(store, SQLModel.metadata)

        # Check that both parent and error tables have prefix
        assert "myprefix_prefixed" in metadata.tables
        assert "myprefix_prefixed__errors" in metadata.tables

        # Unprefixed versions should not exist
        assert "prefixed" not in metadata.tables
        assert "prefixed__errors" not in metadata.tables


def test_error_table_project_filtering():
    """Test that error tables are only included when parent is in the filtered project."""
    config_a = MetaxyConfig(
        project="project_a",
        stores={
            "test_store": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": ":memory:"},
            )
        },
        store="test_store",
        ext={"sqlmodel": SQLModelPluginConfig(enable=True)},
    )

    with config_a.use():

        class ProjectAFeature(
            BaseSQLModelFeature,
            table=True,
            inject_primary_key=False,
            spec=FeatureSpec(
                key=FeatureKey(["proj_a", "feature"]),
                id_columns=["id"],
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            id: str = Field(primary_key=True)
            data: str

        # Create project B feature in its own context
        with MetaxyConfig(project="project_b").use():

            class ProjectBFeature(
                BaseSQLModelFeature,
                table=True,
                inject_primary_key=False,
                spec=FeatureSpec(
                    key=FeatureKey(["proj_b", "feature"]),
                    id_columns=["id"],
                    fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
                ),
            ):
                id: str = Field(primary_key=True)
                data: str

        store = config_a.get_store(expected_type=IbisMetadataStore)

        # Filter for project_a only
        url, metadata = filter_feature_sqlmodel_metadata(store, SQLModel.metadata)

        # Project A feature and error table should be included
        assert "proj_a__feature" in metadata.tables
        assert "proj_a__feature__errors" in metadata.tables

        # Project B feature and error table should NOT be included
        assert "proj_b__feature" not in metadata.tables
        assert "proj_b__feature__errors" not in metadata.tables


def test_error_table_with_composite_id_columns():
    """Test that error tables work correctly with composite (multi-column) id_columns."""
    config = MetaxyConfig(
        project="test_project",
        stores={
            "test_store": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": ":memory:"},
            )
        },
        store="test_store",
        ext={"sqlmodel": SQLModelPluginConfig(enable=True)},
    )

    with config.use():

        class CompositeIdFeature(
            BaseSQLModelFeature,
            table=True,
            inject_primary_key=False,
            spec=FeatureSpec(
                key=FeatureKey(["composite"]),
                id_columns=["user_id", "session_id", "timestamp"],
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            user_id: int = Field(primary_key=True)
            session_id: str = Field(primary_key=True)
            timestamp: int = Field(primary_key=True)
            data: str

        store = config.get_store(expected_type=IbisMetadataStore)
        url, metadata = filter_feature_sqlmodel_metadata(store, SQLModel.metadata)

        # Get error table
        error_table = metadata.tables["composite__errors"]

        # Check all id_columns exist
        assert error_table.c.get("user_id") is not None
        assert error_table.c.get("session_id") is not None
        assert error_table.c.get("timestamp") is not None

        # Check types match parent
        parent_table = metadata.tables["composite"]
        assert (
            type(error_table.c.user_id.type).__name__
            == type(parent_table.c.user_id.type).__name__
        )
        assert (
            type(error_table.c.session_id.type).__name__
            == type(parent_table.c.session_id.type).__name__
        )


def test_error_table_primary_key_injection():
    """Test that error tables get primary key constraints when inject_primary_key is True."""
    config = MetaxyConfig(
        project="test_project",
        stores={
            "test_store": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": ":memory:"},
            )
        },
        store="test_store",
        ext={"sqlmodel": SQLModelPluginConfig(enable=True)},
    )

    with config.use():

        class PKInjectionFeature(
            BaseSQLModelFeature,
            table=True,
            inject_primary_key=True,
            spec=FeatureSpec(
                key=FeatureKey(["pkinjection"]),
                id_columns=["id"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            id: str = Field(primary_key=True)
            value: str

        store = config.get_store(expected_type=IbisMetadataStore)
        url, metadata = filter_feature_sqlmodel_metadata(
            store, SQLModel.metadata, inject_primary_key=True
        )

        # Get error table
        error_table = metadata.tables["pkinjection__errors"]

        # Check for primary key constraint
        pk_constraint = error_table.primary_key
        assert pk_constraint is not None

        # Check that pk includes id_columns + metaxy_created_at + metaxy_data_version
        pk_column_names = [col.name for col in pk_constraint.columns]
        assert "id" in pk_column_names
        assert METAXY_CREATED_AT in pk_column_names
        assert METAXY_DATA_VERSION in pk_column_names


def test_error_table_index_injection():
    """Test that error tables get index constraints when inject_index is True."""
    config = MetaxyConfig(
        project="test_project",
        stores={
            "test_store": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": ":memory:"},
            )
        },
        store="test_store",
        ext={"sqlmodel": SQLModelPluginConfig(enable=True)},
    )

    with config.use():

        class IndexInjectionFeature(
            BaseSQLModelFeature,
            table=True,
            inject_index=True,
            inject_primary_key=False,
            spec=FeatureSpec(
                key=FeatureKey(["idxinjection"]),
                id_columns=["id"],
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            id: str = Field(primary_key=True)
            value: str

        store = config.get_store(expected_type=IbisMetadataStore)
        url, metadata = filter_feature_sqlmodel_metadata(
            store, SQLModel.metadata, inject_index=True, inject_primary_key=False
        )

        # Get error table
        error_table = metadata.tables["idxinjection__errors"]

        # Check for metaxy_idx index
        indexes = list(error_table.indexes)
        metaxy_idx = next((idx for idx in indexes if idx.name == "metaxy_idx"), None)
        assert metaxy_idx is not None

        # Check that index includes id_columns + metaxy_created_at + metaxy_data_version
        idx_column_names = [col.name for col in metaxy_idx.columns]
        assert "id" in idx_column_names
        assert METAXY_CREATED_AT in idx_column_names
        assert METAXY_DATA_VERSION in idx_column_names
