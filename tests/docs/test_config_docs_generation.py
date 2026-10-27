"""Tests for config documentation generation."""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from metaxy.config import MetaxyConfig
from metaxy.ext.polars.handlers.delta import DeltaMetadataStore, DeltaMetadataStoreConfig
from metaxy_testing.config import SamplePluginConfig
from mkdocs_metaxy.config_generator import (
    extract_field_info,
    format_toml_value,
    generate_config_tabs,
    generate_field_tabs,
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


def test_extract_field_info_plugin_config() -> None:
    """Test extracting field info from PluginConfig."""
    from metaxy.config import PluginConfig

    fields = extract_field_info(PluginConfig)

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
    from metaxy.ext.duckdb import PostgresCatalogConfig

    fields = extract_field_info(PostgresCatalogConfig)

    type_field = next(f for f in fields if f["name"] == "type")
    assert type_field["is_discriminator"] is True


def test_extract_field_info_nested_model_detection() -> None:
    """Test that nested pydantic model fields are detected."""
    from metaxy.ext.duckdb import DuckDBMetadataStoreConfig

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


def test_generated_required_store_field_env_placeholder() -> None:
    field = next(field for field in extract_field_info(DeltaMetadataStoreConfig) if field["name"] == "root_path")
    tabs = generate_field_tabs(field, ["stores", "dev", "config"], env_prefix="METAXY_")
    export_line = next(line.strip() for line in tabs.splitlines() if line.strip().startswith("export "))
    _, assignment = shlex.split(export_line)
    name, value = assignment.split("=", 1)

    assert name == "METAXY_STORES__DEV__CONFIG"
    assert json.loads(value) == {"root_path": "..."}


@pytest.fixture
def apply_generated_env(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    def apply(tabs: str) -> None:
        for line in tabs.splitlines():
            if line.strip().startswith("export "):
                _, assignment = shlex.split(line.strip())
                name, value = assignment.split("=", 1)
                monkeypatch.setenv(name, value)

    return apply


@pytest.fixture
def delta_doc_config(tmp_path: Path) -> Path:
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text(
        f'''[stores.dev]
type = "metaxy.ext.polars.handlers.delta.DeltaMetadataStore"
[stores.dev.config]
root_path = "{(tmp_path / "original").as_posix()}"
fallback_stores = ["prod"]
''',
        encoding="utf-8",
    )
    return config_file


@pytest.mark.parametrize(
    ("field_name", "example_value"),
    [
        pytest.param("fallback_stores", [], id="list"),
        pytest.param("storage_options", {"token": "it's a token"}, id="dict"),
        pytest.param("root_path", "123", id="scalar"),
    ],
)
def test_generated_store_field_env_example(
    delta_doc_config: Path,
    tmp_path: Path,
    apply_generated_env: Callable[[str], None],
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    example_value: Any,
) -> None:
    monkeypatch.chdir(tmp_path)
    field = next(field for field in extract_field_info(DeltaMetadataStoreConfig) if field["name"] == field_name)
    field["default"] = example_value
    apply_generated_env(generate_field_tabs(field, ["stores", "dev", "config"], env_prefix="METAXY_"))

    store = MetaxyConfig.load(delta_doc_config).get_store("dev", expected_type=DeltaMetadataStore)
    expected: dict[str, Any] = {"root_path": "original", "fallback_stores": ["prod"], "storage_options": {}}
    expected[field_name] = example_value
    assert store._root_uri == str(tmp_path / expected["root_path"])
    assert store.fallback_stores == expected["fallback_stores"]
    assert store.storage_options == expected["storage_options"]


def test_generated_store_config_env_example(
    delta_doc_config: Path,
    tmp_path: Path,
    apply_generated_env: Callable[[str], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    examples = {"root_path": "123", "fallback_stores": [], "storage_options": {"token": "it's a token"}}
    fields = [field for field in extract_field_info(DeltaMetadataStoreConfig) if field["name"] in examples]
    for field in fields:
        field["default"] = examples[field["name"]]
    apply_generated_env(generate_config_tabs(fields, ["stores", "dev", "config"], env_prefix="METAXY_"))

    store = MetaxyConfig.load(delta_doc_config).get_store("dev", expected_type=DeltaMetadataStore)

    assert store._root_uri == str(tmp_path / "123")
    assert store.fallback_stores == []
    assert store.storage_options == {"token": "it's a token"}


def test_generated_nested_store_env_example(apply_generated_env: Callable[[str], None]) -> None:
    from metaxy.ext.duckdb import DuckDBMetadataStore, DuckDBMetadataStoreConfig

    field = next(
        field
        for field in extract_field_info(DuckDBMetadataStoreConfig)
        if field["path"] == ["ducklake", "attach_options"]
    )
    field["default"] = {"read_only": True}
    apply_generated_env(generate_field_tabs(field, ["stores", "dev", "config"], env_prefix="METAXY_"))
    config = MetaxyConfig(
        stores={
            "dev": {
                "type": DuckDBMetadataStore,
                "config": {
                    "database": ":memory:",
                    "ducklake": {"catalog": {"type": "motherduck", "database": "lake"}},
                },
            }
        }
    )

    store = config.get_store("dev", expected_type=DuckDBMetadataStore)

    assert store.ducklake_attachment_config.attach_options == {"read_only": True}
