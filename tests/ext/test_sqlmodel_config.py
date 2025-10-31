"""Tests for SQLModel configuration-controlled auto table naming.

This module tests that the infer_db_table_names configuration option
can be controlled through various methods:
1. Explicit config setting
2. TOML file configuration
3. Environment variable configuration
"""

import os
from collections.abc import Sequence

from sqlmodel import Field

from metaxy import BaseFeatureSpec, FeatureKey, FieldKey, FieldSpec, TestingFeatureSpec
from metaxy._testing import TempMetaxyProject
from metaxy.config import ExtConfig, MetaxyConfig, SQLModelConfig
from metaxy.ext.sqlmodel import BaseSQLModelFeature
from metaxy.models.feature import FeatureGraph


class FeatureSpec(BaseFeatureSpec[Sequence[str]]):
    id_columns: Sequence[str] = ["sample_id"]


class SQLModelFeature(BaseSQLModelFeature):
    sample_id: str


def test_auto_table_naming_enabled_by_default():
    """Test that auto table naming is enabled by default."""
    # Clear any existing config
    MetaxyConfig.reset()

    test_graph = FeatureGraph()

    with test_graph.use():

        class TestFeature(
            SQLModelFeature,
            table=True,
            spec=TestingFeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version=1)],
            ),
        ):
            # No __tablename__ specified
            uid: str = Field(primary_key=True)
            value: str

        # Should automatically be set to "test__feature"
        tablename = str(TestFeature.__tablename__)  # type: ignore[arg-type]
        assert tablename == "test__feature"


def test_auto_table_naming_disabled_explicit_config():
    """Test disabling auto table naming through explicit config setting."""
    # Create config with infer_db_table_names disabled
    config = MetaxyConfig(
        ext=ExtConfig(sqlmodel=SQLModelConfig(infer_db_table_names=False))
    )
    MetaxyConfig.set(config)

    test_graph = FeatureGraph()

    with test_graph.use():

        class DisabledAutoNaming(
            SQLModelFeature,
            table=True,
            spec=TestingFeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version=1)],
            ),
        ):
            # No __tablename__ specified - should NOT use feature key table name
            uid: str = Field(primary_key=True)
            value: str

        # Should NOT have the auto-generated name from feature key
        # SQLModel will use its default (class name lowercased)
        tablename = str(DisabledAutoNaming.__tablename__)  # type: ignore[arg-type]
        assert tablename != "test__feature"
        # SQLModel's default is the class name lowercased
        assert tablename == "disabledautonaming"

    # Reset config
    MetaxyConfig.reset()


def test_auto_table_naming_still_allows_explicit():
    """Test that explicit __tablename__ works regardless of config."""
    # Test with auto naming disabled
    config = MetaxyConfig(
        ext=ExtConfig(sqlmodel=SQLModelConfig(infer_db_table_names=False))
    )
    MetaxyConfig.set(config)

    test_graph = FeatureGraph()

    with test_graph.use():

        class ExplicitTableName(
            SQLModelFeature,
            table=True,
            spec=TestingFeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version=1)],
            ),
        ):
            __tablename__: str = "custom_table"  # pyright: ignore[reportIncompatibleVariableOverride]
            uid: str = Field(primary_key=True)
            value: str

        # Explicit __tablename__ should work regardless of config
        assert ExplicitTableName.__tablename__ == "custom_table"

    # Reset config
    MetaxyConfig.reset()


def test_auto_table_naming_toml_config(tmp_path):
    """Test controlling auto table naming through TOML file."""
    # Create a TempMetaxyProject with custom config
    config_content = """
[ext.sqlmodel]
infer_db_table_names = false

[stores.dev]
type = "metaxy.metadata_store.memory.InMemoryMetadataStore"
"""
    project = TempMetaxyProject(tmp_path, config_content=config_content)

    # Load config from TOML
    config = MetaxyConfig.load(config_file=project.project_dir / "metaxy.toml")
    assert config.ext.sqlmodel.infer_db_table_names is False
    MetaxyConfig.set(config)

    test_graph = FeatureGraph()

    with test_graph.use():

        class TomlConfigFeature(
            SQLModelFeature,
            table=True,
            spec=TestingFeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version=1)],
            ),
        ):
            uid: str = Field(primary_key=True)
            value: str

        # Should NOT have the auto-generated name from feature key
        tablename = str(TomlConfigFeature.__tablename__)  # type: ignore[arg-type]
        assert tablename != "test__feature"
        # SQLModel's default is the class name lowercased
        assert tablename == "tomlconfigfeature"

    # Reset config
    MetaxyConfig.reset()


def test_auto_table_naming_env_var():
    """Test controlling auto table naming through environment variable."""
    # Save original env var if it exists
    original_value = os.environ.get("METAXY_EXT__SQLMODEL__INFER_DB_TABLE_NAMES")

    try:
        # Set env var to disable auto table naming
        os.environ["METAXY_EXT__SQLMODEL__INFER_DB_TABLE_NAMES"] = "false"

        # Create config - env var should override default
        config = MetaxyConfig()
        assert config.ext.sqlmodel.infer_db_table_names is False
        MetaxyConfig.set(config)

        test_graph = FeatureGraph()

        with test_graph.use():

            class EnvVarFeature(
                SQLModelFeature,
                table=True,
                spec=TestingFeatureSpec(
                    key=FeatureKey(["test", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["data"]), code_version=1)],
                ),
            ):
                uid: str = Field(primary_key=True)
                value: str

            # Should NOT have the auto-generated name from feature key
            tablename = str(EnvVarFeature.__tablename__)  # type: ignore[arg-type]
            assert tablename != "test__feature"
            # SQLModel's default is the class name lowercased
            assert tablename == "envvarfeature"

    finally:
        # Restore original env var
        if original_value is not None:
            os.environ["METAXY_EXT__SQLMODEL__INFER_DB_TABLE_NAMES"] = original_value
        else:
            os.environ.pop("METAXY_EXT__SQLMODEL__INFER_DB_TABLE_NAMES", None)

        # Reset config
        MetaxyConfig.reset()


def test_config_priority_init_over_env():
    """Test that init arguments have priority over environment variables."""
    # Save original env var if it exists
    original_value = os.environ.get("METAXY_EXT__SQLMODEL__INFER_DB_TABLE_NAMES")

    try:
        # Set env var to enable auto table naming
        os.environ["METAXY_EXT__SQLMODEL__INFER_DB_TABLE_NAMES"] = "true"

        # Create config with explicit init arg to disable
        config = MetaxyConfig(
            ext=ExtConfig(sqlmodel=SQLModelConfig(infer_db_table_names=False))
        )

        # Init arg should win over env var
        assert config.ext.sqlmodel.infer_db_table_names is False
        MetaxyConfig.set(config)

        test_graph = FeatureGraph()

        with test_graph.use():

            class PriorityFeature(
                SQLModelFeature,
                table=True,
                spec=TestingFeatureSpec(
                    key=FeatureKey(["test", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["data"]), code_version=1)],
                ),
            ):
                uid: str = Field(primary_key=True)
                value: str

            # Should NOT have the auto-generated name from feature key
            tablename = str(PriorityFeature.__tablename__)  # type: ignore[arg-type]
            assert tablename != "test__feature"
            # SQLModel's default is the class name lowercased
            assert tablename == "priorityfeature"

    finally:
        # Restore original env var
        if original_value is not None:
            os.environ["METAXY_EXT__SQLMODEL__INFER_DB_TABLE_NAMES"] = original_value
        else:
            os.environ.pop("METAXY_EXT__SQLMODEL__INFER_DB_TABLE_NAMES", None)

        # Reset config
        MetaxyConfig.reset()


def test_pyproject_toml_config(tmp_path):
    """Test loading config from pyproject.toml [tool.metaxy] section."""
    # Create a pyproject.toml with metaxy config
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text("""
[tool.metaxy.ext.sqlmodel]
infer_db_table_names = false

[tool.metaxy.stores.dev]
type = "metaxy.metadata_store.memory.InMemoryMetadataStore"
""")

    # Load config from pyproject.toml
    config = MetaxyConfig.load(config_file=config_path)
    assert config.ext.sqlmodel.infer_db_table_names is False
    MetaxyConfig.set(config)

    test_graph = FeatureGraph()

    with test_graph.use():

        class PyProjectFeature(
            SQLModelFeature,
            table=True,
            spec=TestingFeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version=1)],
            ),
        ):
            uid: str = Field(primary_key=True)
            value: str

        # Should NOT have the auto-generated name from feature key
        tablename = str(PyProjectFeature.__tablename__)  # type: ignore[arg-type]
        assert tablename != "test__feature"
        # SQLModel's default is the class name lowercased
        assert tablename == "pyprojectfeature"

    # Reset config
    MetaxyConfig.reset()


def test_multiple_features_with_config_change():
    """Test that config changes affect new features correctly."""
    test_graph = FeatureGraph()

    # Start with auto naming enabled
    config1 = MetaxyConfig(
        ext=ExtConfig(sqlmodel=SQLModelConfig(infer_db_table_names=True))
    )
    MetaxyConfig.set(config1)

    with test_graph.use():

        class Feature1(
            SQLModelFeature,
            table=True,
            spec=TestingFeatureSpec(
                key=FeatureKey(["feature", "one"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version=1)],
            ),
        ):
            uid: str = Field(primary_key=True)

        # Should have auto-generated name
        tablename = str(Feature1.__tablename__)  # type: ignore[arg-type]
        assert tablename == "feature__one"

        # Now disable auto naming
        config2 = MetaxyConfig(
            ext=ExtConfig(sqlmodel=SQLModelConfig(infer_db_table_names=False))
        )
        MetaxyConfig.set(config2)

        # New feature should not get auto-generated name
        class Feature2(
            SQLModelFeature,
            table=True,
            spec=TestingFeatureSpec(
                key=FeatureKey(["feature", "two"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version=1)],
            ),
        ):
            uid: str = Field(primary_key=True)

        # Should NOT have the auto-generated name from feature key
        tablename = str(Feature2.__tablename__)  # type: ignore[arg-type]
        assert tablename != "feature__two"
        # SQLModel's default is the class name lowercased
        assert tablename == "feature2"

    # Reset config
    MetaxyConfig.reset()
