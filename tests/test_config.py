"""Tests for configuration system."""

from pathlib import Path

import pytest

from metaxy.config import (
    InvalidConfigError,
    MetaxyConfig,
    StoreConfig,
    _collect_dict_keys,
)
from metaxy.metadata_store.duckdb import DuckDBMetadataStore


class TestCollectDictKeys:
    """Tests for _collect_dict_keys helper function."""

    def test_empty_dict(self) -> None:
        assert _collect_dict_keys({}) == []

    def test_flat_dict(self) -> None:
        result = _collect_dict_keys({"a": 1, "b": 2})
        assert set(result) == {"a", "b"}

    def test_nested_dict(self) -> None:
        result = _collect_dict_keys({"config": {"host": "localhost", "port": 5432}})
        assert set(result) == {"config", "config.host", "config.port"}

    def test_deeply_nested_dict(self) -> None:
        result = _collect_dict_keys({"a": {"b": {"c": {"d": 1}}}})
        assert set(result) == {"a", "a.b", "a.b.c", "a.b.c.d"}

    def test_dict_with_list_value(self) -> None:
        """Lists should not be recursed into."""
        result = _collect_dict_keys({"items": [1, 2, 3], "name": "test"})
        assert set(result) == {"items", "name"}

    def test_dict_with_none_value(self) -> None:
        result = _collect_dict_keys({"value": None, "other": "test"})
        assert set(result) == {"value", "other"}

    def test_dict_with_mixed_values(self) -> None:
        result = _collect_dict_keys(
            {
                "string": "hello",
                "number": 42,
                "nested": {"inner": "value"},
                "list": [1, 2],
                "none": None,
            }
        )
        assert set(result) == {
            "string",
            "number",
            "nested",
            "nested.inner",
            "list",
            "none",
        }


def test_store_config_basic() -> None:
    from metaxy import DuckDBMetadataStore

    config = StoreConfig(
        type="metaxy.metadata_store.DuckDBMetadataStore",
        config={},
    )

    assert config.type == DuckDBMetadataStore
    assert config.config == {}


def test_store_config_with_options() -> None:
    from metaxy import DuckDBMetadataStore

    config = StoreConfig(
        type="metaxy.metadata_store.DuckDBMetadataStore",
        config={
            "fallback_stores": ["prod"],
        },
    )

    assert config.type == DuckDBMetadataStore
    assert config.config["fallback_stores"] == ["prod"]


def test_metaxy_config_default() -> None:
    config = MetaxyConfig()

    assert config.store == "dev"
    assert config.stores == {}


def test_metaxy_config_from_dict() -> None:
    config = MetaxyConfig(
        store="staging",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={},
            ),
            "staging": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={"fallback_stores": ["prod"]},
            ),
            "prod": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={},
            ),
        },
    )

    assert config.store == "staging"
    assert len(config.stores) == 3
    assert "dev" in config.stores
    assert "staging" in config.stores
    assert "prod" in config.stores


def test_load_from_metaxy_toml(tmp_path: Path) -> None:
    from metaxy import DuckDBMetadataStore

    # Create metaxy.toml
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.DuckDBMetadataStore"

[stores.dev.config]
# No config needed for in-memory

[stores.prod]
type = "metaxy.metadata_store.DuckDBMetadataStore"

[stores.prod.config]
fallback_stores = []
""")

    # Load config
    config = MetaxyConfig.load(config_file)

    assert config.store == "dev"
    assert len(config.stores) == 2
    assert config.stores["dev"].type == DuckDBMetadataStore
    assert config.stores["prod"].type == DuckDBMetadataStore


def test_load_from_pyproject_toml(tmp_path: Path) -> None:
    config_file = tmp_path / "pyproject.toml"
    config_file.write_text("""
[project]
name = "test"

[tool.metaxy]
store = "staging"

[tool.metaxy.stores.staging]
type = "metaxy.metadata_store.DuckDBMetadataStore"

[tool.metaxy.stores.staging.config]
fallback_stores = ["prod"]

[tool.metaxy.stores.prod]
type = "metaxy.metadata_store.DuckDBMetadataStore"

[tool.metaxy.stores.prod.config]
""")

    config = MetaxyConfig.load(config_file)

    assert config.store == "staging"
    assert len(config.stores) == 2
    assert config.stores["staging"].config["fallback_stores"] == ["prod"]


def test_load_from_metaxy_config_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that METAXY_CONFIG env var is respected for config file location."""
    from metaxy import DuckDBMetadataStore

    # Create config file in a non-standard location
    config_dir = tmp_path / "custom" / "config" / "location"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "my-metaxy-config.toml"
    config_file.write_text("""
project = "env_var_project"
store = "custom"

[stores.custom]
type = "metaxy.metadata_store.DuckDBMetadataStore"
""")

    # Set the env var to point to our config file
    monkeypatch.setenv("METAXY_CONFIG", str(config_file))

    # Load config without specifying file - should use env var
    config = MetaxyConfig.load()

    assert config.project == "env_var_project"
    assert config.store == "custom"
    assert config.stores["custom"].type == DuckDBMetadataStore
    assert config.config_file == config_file.resolve()


def test_init_metaxy_respects_metaxy_config_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that init_metaxy respects METAXY_CONFIG env var."""
    from metaxy import init_metaxy

    # Create config file with entrypoints
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
project = "init_test_project"
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.DuckDBMetadataStore"
""")

    # Set the env var
    monkeypatch.setenv("METAXY_CONFIG", str(config_file))

    # Call init_metaxy without specifying config file
    config = init_metaxy()

    assert config.project == "init_test_project"
    assert config.config_file == config_file.resolve()


def test_get_store_instantiates_correctly() -> None:
    config = MetaxyConfig(
        store="dev",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={},
            )
        },
    )

    store = config.get_store("dev")

    assert isinstance(store)
    assert store.fallback_stores == []


def test_get_store_with_fallback_chain() -> None:
    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={"fallback_stores": ["staging"]},
            ),
            "staging": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={"fallback_stores": ["prod"]},
            ),
            "prod": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={},
            ),
        },
    )

    dev_store = config.get_store("dev")

    assert isinstance(dev_store)
    assert len(dev_store.fallback_stores) == 1

    staging_store = dev_store.fallback_stores[0]
    assert isinstance(staging_store)
    assert len(staging_store.fallback_stores) == 1

    prod_store = staging_store.fallback_stores[0]
    assert isinstance(prod_store)
    assert len(prod_store.fallback_stores) == 0


def test_get_store_uses_default() -> None:
    config = MetaxyConfig(
        store="staging",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={},
            ),
            "staging": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={},
            ),
        },
    )

    # Without name, should use store
    store = config.get_store()
    assert isinstance(store)

    # Verify it's actually staging by checking it has no special config
    # (would need better verification in real scenario)


def test_get_store_nonexistent_raises() -> None:
    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={},
            )
        }
    )

    with pytest.raises(InvalidConfigError, match="Store 'nonexistent' not found"):
        config.get_store("nonexistent")


def test_config_with_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from metaxy import DuckDBMetadataStore

    # Create config file
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.DuckDBMetadataStore"

[stores.dev.config]
""")

    # Override store via env var
    monkeypatch.setenv("METAXY_STORE", "prod")

    # Also add prod store via env var (pydantic-settings supports this!)
    monkeypatch.setenv(
        "METAXY_STORES__PROD__TYPE", "metaxy.metadata_store.DuckDBMetadataStore"
    )

    config = MetaxyConfig.load(config_file)

    # Env var should override TOML
    assert config.store == "prod"

    # Store from env var should be available
    assert "prod" in config.stores
    assert config.stores["prod"].type == DuckDBMetadataStore


def test_partial_env_var_store_config_filtered_out_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that partial env vars for a store are filtered out with a warning.

    When env vars like METAXY_STORES__PROD__CONFIG__CONNECTION_STRING are set
    without METAXY_STORES__PROD__TYPE, the incomplete store config should be
    filtered out and a warning emitted that includes the fields that were set.
    """
    # Create config file with only dev store
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.DuckDBMetadataStore"
""")

    # Set partial config for prod store (config but no type)
    monkeypatch.setenv(
        "METAXY_STORES__PROD__CONFIG__CONNECTION_STRING", "clickhouse://localhost"
    )

    # Config should load successfully but emit a warning with field hints
    with pytest.warns(
        UserWarning,
        match=r"Ignoring incomplete store config 'prod'.*has fields: config.*environment variables",
    ):
        config = MetaxyConfig.load(config_file)

    # dev store should work
    assert config.stores["dev"].type_path == "metaxy.metadata_store.DuckDBMetadataStore"

    # prod store should be filtered out (not present) since it lacks 'type'
    assert "prod" not in config.stores


def test_incomplete_store_warning_shows_all_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that the warning for incomplete stores lists all fields that were set."""
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.DuckDBMetadataStore"
""")

    # Set multiple fields for an incomplete store
    monkeypatch.setenv(
        "METAXY_STORES__PROD__CONFIG__CONNECTION_STRING", "clickhouse://localhost"
    )
    monkeypatch.setenv("METAXY_STORES__PROD__CONFIG__DATABASE", "mydb")

    with pytest.warns(UserWarning) as record:
        MetaxyConfig.load(config_file)

    # Find the warning for 'prod' store
    prod_warnings = [w for w in record if "prod" in str(w.message)]
    assert len(prod_warnings) == 1

    warning_msg = str(prod_warnings[0].message)
    assert "missing required 'type' field" in warning_msg
    assert "has fields:" in warning_msg
    # Should show nested fields with dot notation
    assert "config.connection_string" in warning_msg
    assert "config.database" in warning_msg
    assert "environment variables" in warning_msg


def test_hash_algorithm_must_match_in_fallback_chain() -> None:
    from metaxy.versioning.types import HashAlgorithm

    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={
                    "hash_algorithm": "sha256",
                    "fallback_stores": ["staging"],
                },
            ),
            "staging": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={"hash_algorithm": "sha256", "fallback_stores": ["prod"]},
            ),
            "prod": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={"hash_algorithm": "sha256"},
            ),
        },
    )

    dev_store = config.get_store("dev")

    # All stores should use the same algorithm
    assert dev_store.hash_algorithm == HashAlgorithm.SHA256

    staging_store = dev_store.fallback_stores[0]
    assert staging_store.hash_algorithm == HashAlgorithm.SHA256

    prod_store = staging_store.fallback_stores[0]
    assert prod_store.hash_algorithm == HashAlgorithm.SHA256


def test_hash_algorithm_defaults_to_xxhash64() -> None:
    from metaxy.versioning.types import HashAlgorithm

    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={"fallback_stores": ["prod"]},
            ),
            "prod": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={},
            ),
        },
    )

    dev_store = config.get_store("dev")

    # Should default to XXHASH64
    assert dev_store.hash_algorithm == HashAlgorithm.XXHASH64

    # Fallback should also use XXHASH64
    prod_store = dev_store.fallback_stores[0]
    assert prod_store.hash_algorithm == HashAlgorithm.XXHASH64


def test_hash_algorithm_conflict_raises_error() -> None:
    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={
                    "hash_algorithm": "sha256",
                    "fallback_stores": ["prod"],
                },
            ),
            "prod": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={"hash_algorithm": "md5"},
            ),
        },
    )

    # Hash algorithm conflict is checked when store is opened
    with pytest.raises(
        ValueError,
        match="Fallback store 0 uses hash_algorithm='md5' but this store uses 'sha256'",
    ):
        dev_store = config.get_store("dev")
        with dev_store:
            pass  # Error raised on __enter__


def test_store_respects_configured_hash_algorithm() -> None:
    from metaxy.versioning.types import HashAlgorithm

    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={"hash_algorithm": "md5"},
            ),
        },
    )

    store = config.get_store("dev")

    # Store should use the configured algorithm
    assert store.hash_algorithm == HashAlgorithm.MD5


def test_env_var_expansion_in_toml_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that ${VAR} syntax expands environment variables in TOML config."""
    # Set environment variables
    monkeypatch.setenv("DAGSTER_CLOUD_GIT_BRANCH", "feature-branch")
    monkeypatch.setenv("MY_PROJECT_NAME", "test-project")

    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
store = "dev"
project = "${MY_PROJECT_NAME}"

[stores.dev]
type = "metaxy.metadata_store.DuckDBMetadataStore"

[stores.dev.config]
branch = "${DAGSTER_CLOUD_GIT_BRANCH}"
""")

    config = MetaxyConfig.load(config_file)

    assert config.project == "test-project"
    assert config.stores["dev"].config["branch"] == "feature-branch"


def test_env_var_expansion_with_default_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that ${VAR:-default} syntax uses default when env var is not set."""
    # Ensure the variable is NOT set
    monkeypatch.delenv("UNSET_VAR", raising=False)
    # Set one variable to verify it takes precedence over default
    monkeypatch.setenv("SET_VAR", "actual-value")

    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
store = "dev"
project = "${UNSET_VAR:-default-project}"

[stores.dev]
type = "metaxy.metadata_store.DuckDBMetadataStore"

[stores.dev.config]
value_with_default = "${SET_VAR:-fallback}"
unset_with_default = "${ANOTHER_UNSET:-my-default}"
""")

    config = MetaxyConfig.load(config_file)

    assert config.project == "default-project"
    assert config.stores["dev"].config["value_with_default"] == "actual-value"
    assert config.stores["dev"].config["unset_with_default"] == "my-default"


def test_env_var_expansion_unset_becomes_empty_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that ${VAR} without default becomes empty string when unset."""
    monkeypatch.delenv("UNSET_VAR", raising=False)

    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
store = "dev"
migrations_dir = "prefix-${UNSET_VAR}-suffix"

[stores.dev]
type = "metaxy.metadata_store.DuckDBMetadataStore"
""")

    config = MetaxyConfig.load(config_file)

    assert config.migrations_dir == "prefix--suffix"


def test_env_var_expansion_in_nested_structures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that env vars are expanded in nested dicts and lists."""
    monkeypatch.setenv("STORE_PATH", "/data/metadata")
    monkeypatch.setenv("FALLBACK_STORE", "prod")

    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
store = "dev"
entrypoints = ["${STORE_PATH}/features", "another/path"]

[stores.dev]
type = "metaxy.metadata_store.DuckDBMetadataStore"

[stores.dev.config]
path = "${STORE_PATH}"
fallback_stores = ["${FALLBACK_STORE}"]
""")

    config = MetaxyConfig.load(config_file)

    assert config.entrypoints == ["/data/metadata/features", "another/path"]
    assert config.stores["dev"].config["path"] == "/data/metadata"
    assert config.stores["dev"].config["fallback_stores"] == ["prod"]


def test_get_store_with_fallback_chain_delta(tmp_path: Path) -> None:
    """Test that fallback stores are correctly attached for Delta stores via config.

    This tests the specific issue where get_store() was not correctly resolving
    fallback store names to store instances for stores that use from_config().
    """
    from metaxy.metadata_store.delta import DeltaMetadataStore

    dev_path = tmp_path / "dev"
    staging_path = tmp_path / "staging"
    prod_path = tmp_path / "prod"

    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.delta.DeltaMetadataStore",
                config={
                    "root_path": str(dev_path),
                    "fallback_stores": ["staging"],
                },
            ),
            "staging": StoreConfig(
                type="metaxy.metadata_store.delta.DeltaMetadataStore",
                config={
                    "root_path": str(staging_path),
                    "fallback_stores": ["prod"],
                },
            ),
            "prod": StoreConfig(
                type="metaxy.metadata_store.delta.DeltaMetadataStore",
                config={
                    "root_path": str(prod_path),
                },
            ),
        },
    )

    dev_store = config.get_store("dev")

    assert isinstance(dev_store, DeltaMetadataStore)
    assert len(dev_store.fallback_stores) == 1, (
        f"Expected 1 fallback store, got {len(dev_store.fallback_stores)}. "
        f"Fallback stores are not being correctly resolved via config.get_store()."
    )

    staging_store = dev_store.fallback_stores[0]
    assert isinstance(staging_store, DeltaMetadataStore)
    assert len(staging_store.fallback_stores) == 1

    prod_store = staging_store.fallback_stores[0]
    assert isinstance(prod_store, DeltaMetadataStore)
    assert len(prod_store.fallback_stores) == 0


def test_get_store_with_fallback_chain_delta_from_toml(tmp_path: Path) -> None:
    """Test that fallback stores are correctly attached when loading Delta stores from TOML.

    This tests the specific issue where get_store() was not correctly resolving
    fallback store names to store instances when loading from a TOML config file.
    """
    from metaxy.metadata_store.delta import DeltaMetadataStore

    dev_path = tmp_path / "dev"
    branch_path = tmp_path / "branch"

    config_file = tmp_path / "metaxy.toml"
    config_file.write_text(f"""
project = "test-project"

[stores.dev]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"
[stores.dev.config]
root_path = "{dev_path}"
fallback_stores = ["branch"]

[stores.branch]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"
[stores.branch.config]
root_path = "{branch_path}"
""")

    config = MetaxyConfig.load(config_file)
    dev_store = config.get_store("dev")

    assert isinstance(dev_store, DeltaMetadataStore)
    assert len(dev_store.fallback_stores) == 1, (
        f"Expected 1 fallback store, got {len(dev_store.fallback_stores)}. "
        f"Fallback stores are not being correctly resolved via config.get_store() from TOML."
    )

    branch_store = dev_store.fallback_stores[0]
    assert isinstance(branch_store, DeltaMetadataStore)
    assert len(branch_store.fallback_stores) == 0


def test_config_file_attribute_set_when_loaded_from_file(tmp_path: Path) -> None:
    """Test that config_file attribute is set when loading from a TOML file."""
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
project = "test-project"
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.DuckDBMetadataStore"
""")

    config = MetaxyConfig.load(config_file)

    assert config.config_file is not None
    assert config.config_file == config_file.resolve()
    assert config.config_file.name == "metaxy.toml"


def test_config_file_attribute_set_when_auto_discovered(tmp_path: Path) -> None:
    """Test that config_file attribute is set when auto-discovering config."""
    import os

    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
project = "discovered-project"
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.DuckDBMetadataStore"
""")

    # Change to tmp_path so auto-discovery finds the config
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        config = MetaxyConfig.load()

        assert config.config_file is not None
        assert config.config_file == config_file.resolve()
        assert config.project == "discovered-project"
    finally:
        os.chdir(original_cwd)


def test_config_file_attribute_none_when_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that config_file attribute is None when no config file is used."""
    import os

    # Change to an empty directory with no config files
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    original_cwd = os.getcwd()
    try:
        os.chdir(empty_dir)
        # Disable parent search to ensure no config file is found
        config = MetaxyConfig.load(search_parents=False)

        assert config.config_file is None
    finally:
        os.chdir(original_cwd)


def test_config_file_attribute_none_when_created_directly() -> None:
    """Test that config_file attribute is None when config is created directly."""
    config = MetaxyConfig(project="direct-config")

    assert config.config_file is None


def test_get_store_error_includes_config_file_path(tmp_path: Path) -> None:
    """Test that get_store error messages include the config file path."""
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
project = "test-project"
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.DuckDBMetadataStore"
""")

    config = MetaxyConfig.load(config_file)

    # Try to get a non-existent store
    with pytest.raises(InvalidConfigError) as exc_info:
        config.get_store("nonexistent")

    error_message = str(exc_info.value)
    assert "nonexistent" in error_message
    assert str(config_file.resolve()) in error_message
    assert "Config file:" in error_message
    assert "METAXY_" in error_message  # Check env var note is included


def test_get_store_error_no_stores_includes_config_file_path(tmp_path: Path) -> None:
    """Test that 'no stores available' error includes config file path."""
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
project = "test-project"
""")

    config = MetaxyConfig.load(config_file)

    # Try to get a store when none are configured
    with pytest.raises(InvalidConfigError) as exc_info:
        config.get_store("dev")

    error_message = str(exc_info.value)
    assert "No Metaxy stores available" in error_message
    assert str(config_file.resolve()) in error_message
    assert "Config file:" in error_message
    assert "METAXY_" in error_message  # Check env var note is included


def test_get_store_error_without_config_file_no_path_in_message() -> None:
    """Test that get_store error doesn't include config file path when not loaded from file."""
    config = MetaxyConfig(
        project="direct-config",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.DuckDBMetadataStore",
                config={},
            )
        },
    )

    # Try to get a non-existent store
    with pytest.raises(InvalidConfigError) as exc_info:
        config.get_store("nonexistent")

    error_message = str(exc_info.value)
    assert "nonexistent" in error_message
    assert "Config file:" not in error_message
    # Env var note should still be included
    assert "METAXY_" in error_message


def test_get_store_error_from_config_includes_config_file_path(tmp_path: Path) -> None:
    """Test that from_config errors include the config file path."""
    config_file = tmp_path / "metaxy.toml"
    # Use a Delta store with a path that will fail when opening
    config_file.write_text("""
project = "test-project"
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.dev.config]
root_path = "s3://nonexistent-bucket-that-does-not-exist/path"
fallback_stores = ["nonexistent_store"]
""")

    config = MetaxyConfig.load(config_file)

    # This should fail when trying to instantiate the store (fallback store doesn't exist)
    with pytest.raises(InvalidConfigError) as exc_info:
        config.get_store("dev")

    error_message = str(exc_info.value)
    # The error comes from the nested get_store call for the fallback store
    assert "nonexistent_store" in error_message
    assert "not found in config" in error_message
    assert str(config_file.resolve()) in error_message
    assert "Config file:" in error_message
    assert "METAXY_" in error_message


def test_get_store_error_invalid_config_includes_config_file_path(
    tmp_path: Path,
) -> None:
    """Test that config validation errors include the config file path."""
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
project = "test-project"
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"

[stores.dev.config]
# Missing required db_path, but add an invalid field
invalid_field_that_does_not_exist = "should fail validation"
""")

    config = MetaxyConfig.load(config_file)

    # This should fail when trying to validate the config
    with pytest.raises(InvalidConfigError) as exc_info:
        config.get_store("dev")

    error_message = str(exc_info.value)
    assert "Failed to validate config" in error_message
    assert str(config_file.resolve()) in error_message
    assert "Config file:" in error_message
    assert "METAXY_" in error_message


def test_get_store_error_fallback_store_instantiation_fails(tmp_path: Path) -> None:
    """Test error when a fallback store exists but fails to instantiate."""
    config_file = tmp_path / "metaxy.toml"
    # Use a Delta store with a fallback store that has invalid config
    config_file.write_text("""
project = "test-project"
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"

[stores.dev.config]
root_path = "s3://valid-bucket/path"
fallback_stores = ["branch"]

[stores.branch]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"

[stores.branch.config]
# DuckDB requires database field, this will fail validation
invalid_field = "should_fail"
""")

    config = MetaxyConfig.load(config_file)

    # This should fail when trying to instantiate the fallback store
    with pytest.raises(InvalidConfigError) as exc_info:
        config.get_store("dev")

    error_message = str(exc_info.value)
    # Error should mention the failing store
    assert "branch" in error_message or "Failed to validate" in error_message
    assert str(config_file.resolve()) in error_message
    assert "Config file:" in error_message
    assert "METAXY_" in error_message
    # Should NOT have duplicated error messages (no nested InvalidConfigError wrapping)
    # Count occurrences of "Config file:" - should be exactly 1
    assert error_message.count("Config file:") == 1


def test_invalid_config_error_attributes() -> None:
    """Test that InvalidConfigError has the expected attributes."""
    from pathlib import Path

    # Test with config file
    error = InvalidConfigError("Test message", config_file=Path("/path/to/config.toml"))
    assert error.config_file == Path("/path/to/config.toml")
    assert error.base_message == "Test message"
    assert "Test message" in str(error)
    assert "/path/to/config.toml" in str(error)
    assert "METAXY_" in str(error)

    # Test without config file
    error_no_file = InvalidConfigError("Another message")
    assert error_no_file.config_file is None
    assert error_no_file.base_message == "Another message"
    assert "Another message" in str(error_no_file)
    assert "Config file:" not in str(error_no_file)
    assert "METAXY_" in str(error_no_file)


def test_store_config_type_is_lazy() -> None:
    """Test that StoreConfig.type is lazily resolved on first access."""
    # Use a non-existent import path - if type was eagerly resolved,
    # this would raise an ImportError during StoreConfig creation
    metaxy_config = MetaxyConfig(
        stores={
            "bad_store": StoreConfig(
                type="non_existent_package.BadClass",
                config={},
            )
        }
    )

    # Creating the config should succeed (no import yet)
    assert (
        metaxy_config.stores["bad_store"].type_path == "non_existent_package.BadClass"
    )

    # Accessing the store should fail with an error that includes the store name
    with pytest.raises(InvalidConfigError, match="bad_store") as exc_info:
        metaxy_config.get_store("bad_store")

    # Verify the error message contains useful information
    error_message = str(exc_info.value)
    assert "non_existent_package.BadClass" in error_message
    assert "bad_store" in error_message


def test_store_config_accepts_class_directly() -> None:
    """Test that StoreConfig.type accepts a class object directly."""

    # Passing a class directly should work
    config = StoreConfig(
        type=DuckDBMetadataStore,
        config={},
    )

    # The type_path should be the import string
    assert config.type_path == "metaxy.metadata_store.duckdb.DuckDBMetadataStore"

    # Accessing .type should return the same class
    assert config.type is DuckDBMetadataStore


def test_plugins_respect_metaxy_config_env_var_at_import_time(tmp_path: Path) -> None:
    """Test that sqlalchemy and sqlmodel plugins load config via METAXY_CONFIG at import time.

    These plugins use MetaxyConfig.get(load=True) which should auto-load config
    from METAXY_CONFIG when the global config is not set.
    """
    import subprocess
    import sys

    # Create a config file with a specific project name
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
project = "plugin_import_test_project"
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.DuckDBMetadataStore"
""")

    # Create a test script that imports the plugins and checks config
    test_script = tmp_path / "test_plugin_import.py"
    test_script.write_text("""
import sys

# Reset any existing config state
from metaxy.config import MetaxyConfig
MetaxyConfig.reset()

# Verify config is not set before calling plugin functions
assert not MetaxyConfig.is_set(), "Config should not be set before plugin call"

# Import and call sqlalchemy plugin function that uses MetaxyConfig.get(load=True)
from metaxy.ext.sqlalchemy.plugin import _get_features_metadata
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from sqlalchemy import MetaData

store = DuckDBMetadataStore()

# This should trigger MetaxyConfig.get(load=True) and auto-load from METAXY_CONFIG
_get_features_metadata(source_metadata=MetaData(), store=store)

# Now config should be loaded
assert MetaxyConfig.is_set(), "Config should be set after plugin call"
config = MetaxyConfig.get()
assert config.project == "plugin_import_test_project", f"Expected project 'plugin_import_test_project', got '{config.project}'"

print("SUCCESS: Plugin correctly loaded config from METAXY_CONFIG")
""")

    # Run the test script in a subprocess with METAXY_CONFIG set
    result = subprocess.run(
        [sys.executable, str(test_script)],
        env={**dict(__import__("os").environ), "METAXY_CONFIG": str(config_file)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"Script failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "SUCCESS" in result.stdout
