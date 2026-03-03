"""Tests for config documentation generation."""

from __future__ import annotations

from metaxy_testing.config import SamplePluginConfig
from mkdocs_metaxy.config_generator import (
    extract_field_info,
    format_toml_value,
    generate_config_tabs,
    get_env_var_name,
    get_toml_path,
)


def test_extract_field_info_metaxy_config() -> None:
    """Test extracting field info from MetaxyConfig."""
    from metaxy.config import MetaxyConfig

    fields = extract_field_info(MetaxyConfig)

    assert len(fields) > 0

    store_fields = [f for f in fields if f["name"] == "store"]
    assert len(store_fields) == 1
    assert store_fields[0]["default"] == "dev"


def test_extract_field_info_sqlmodel_config() -> None:
    """Test extracting field info from SQLModelPluginConfig."""
    from metaxy.ext.sqlmodel import SQLModelPluginConfig

    fields = extract_field_info(SQLModelPluginConfig)

    assert len(fields) > 0

    enable_fields = [f for f in fields if f["name"] == "enable"]
    assert len(enable_fields) == 1


def test_extract_field_info_required_detection() -> None:
    """Test that required fields are correctly detected."""
    from metaxy.config import MetaxyConfig

    fields = extract_field_info(MetaxyConfig)

    store_field = next(f for f in fields if f["name"] == "store")
    assert store_field["required"] is False
    assert store_field["default"] == "dev"


def test_extract_field_info_discriminator_detection() -> None:
    """Test that single-value Literal discriminator fields are detected."""
    from metaxy.ext.metadata_stores.ducklake import PostgresCatalogConfig

    fields = extract_field_info(PostgresCatalogConfig)

    type_field = next(f for f in fields if f["name"] == "type")
    assert type_field["is_discriminator"] is True


def test_extract_field_info_nested_model_detection() -> None:
    """Test that nested pydantic model fields are detected."""
    from metaxy.ext.metadata_stores.duckdb import DuckDBMetadataStoreConfig

    fields = extract_field_info(DuckDBMetadataStoreConfig)

    ducklake_fields = [f for f in fields if f["name"] == "ducklake"]
    assert len(ducklake_fields) == 1
    assert ducklake_fields[0]["is_nested"] is True


def test_sample_plugin_config_structure() -> None:
    """Test that SamplePluginConfig generates correct structure."""
    fields = extract_field_info(SamplePluginConfig)

    field_names = {f["name"] for f in fields if not f["is_nested"]}
    assert "enable" in field_names
    assert "name" in field_names
    assert "port" in field_names
    assert "debug" in field_names
    assert "optional_setting" in field_names


def test_format_toml_value() -> None:
    """Test TOML value formatting."""
    assert format_toml_value(True) == "true"
    assert format_toml_value(False) == "false"
    assert format_toml_value("hello") == '"hello"'
    assert format_toml_value(42) == "42"
    assert format_toml_value(3.14) == "3.14"
    assert format_toml_value([]) == "[]"
    assert format_toml_value([1, 2]) == "[1, 2]"
    assert format_toml_value({}) == "{}"


def test_get_env_var_name() -> None:
    """Test environment variable name generation."""
    assert get_env_var_name(["store"], env_prefix="METAXY_") == "METAXY_STORE"
    assert get_env_var_name(["config", "uri"], env_prefix="METAXY_", env_nested_delimiter="__") == "METAXY_CONFIG__URI"


def test_get_toml_path() -> None:
    """Test TOML path generation."""
    assert get_toml_path(["store"]) == "store"
    assert get_toml_path(["config", "uri"]) == "config.uri"


def test_generate_config_tabs_basic() -> None:
    """Test basic config tabs generation."""
    fields = extract_field_info(SamplePluginConfig)
    leaf_fields = [f for f in fields if not f["is_nested"] and not f["is_discriminator"]]

    tabs = generate_config_tabs(leaf_fields, path_prefix=[], env_prefix="SAMPLE_PLUGIN_", env_nested_delimiter="__")

    assert tabs  # Not empty
    assert '=== "metaxy.toml"' in tabs
    assert '=== "pyproject.toml"' in tabs
    assert '=== "Environment Variable"' in tabs
    assert "```toml" in tabs
    assert "```bash" in tabs


def test_generate_config_tabs_with_path_prefix() -> None:
    """Test config tabs generation with a path prefix."""
    fields = extract_field_info(SamplePluginConfig)
    leaf_fields = [f for f in fields if not f["is_nested"] and not f["is_discriminator"]]

    tabs = generate_config_tabs(
        leaf_fields, path_prefix=["stores", "dev", "config"], env_prefix="METAXY_", env_nested_delimiter="__"
    )

    assert "[stores.dev.config]" in tabs
    assert "[tool.metaxy.stores.dev.config]" in tabs


def test_generate_config_tabs_env_vars_uppercased() -> None:
    """Test that environment variable names are uppercased."""
    fields = extract_field_info(SamplePluginConfig)
    enable_field = [f for f in fields if f["name"] == "enable"]

    tabs = generate_config_tabs(enable_field, path_prefix=[], env_prefix="SAMPLE_PLUGIN_", env_nested_delimiter="__")

    assert "SAMPLE_PLUGIN_ENABLE" in tabs


def test_generate_config_tabs_empty_fields() -> None:
    """Test that empty field list returns empty string."""
    tabs = generate_config_tabs([], path_prefix=[], env_prefix="METAXY_", env_nested_delimiter="__")

    assert tabs == ""


def test_generate_config_tabs_no_commented_values() -> None:
    """Test that default values are not commented out in TOML examples."""
    from metaxy.config import MetaxyConfig

    fields = extract_field_info(MetaxyConfig)
    store_field = [f for f in fields if f["name"] == "store"]

    tabs = generate_config_tabs(store_field, path_prefix=[], env_prefix="METAXY_", env_nested_delimiter="__")

    assert 'store = "dev"' in tabs
    assert '# store = "dev"' not in tabs


def test_plugin_directive_pattern() -> None:
    """Test that the directive regex matches correctly."""
    from mkdocs_metaxy.config.plugin import DIRECTIVE_PATTERN

    markdown = """::: metaxy-config
    class: metaxy.config.MetaxyConfig
    header_level: 3
    exclude_fields: stores,ext

Some text after.
"""
    match = DIRECTIVE_PATTERN.search(markdown)
    assert match is not None

    content = match.group(1)
    assert "class: metaxy.config.MetaxyConfig" in content
    assert "header_level: 3" in content


def test_plugin_directive_pattern_no_match() -> None:
    """Test that non-directive content is not matched."""
    from mkdocs_metaxy.config.plugin import DIRECTIVE_PATTERN

    assert DIRECTIVE_PATTERN.search("::: metaxy.SomeClass\n") is None
    assert DIRECTIVE_PATTERN.search("regular text\n") is None
