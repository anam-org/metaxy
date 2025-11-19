"""Tests for configuration system."""

from pathlib import Path

import pytest

from metaxy.config import MetaxyConfig, StoreConfig
from metaxy.metadata_store import InMemoryMetadataStore


def test_store_config_basic() -> None:
    """Test basic StoreConfig structure."""
    from metaxy import InMemoryMetadataStore

    config = StoreConfig(
        type="metaxy.metadata_store.InMemoryMetadataStore",
        config={},
    )

    assert config.type == InMemoryMetadataStore
    assert config.config == {}


def test_store_config_with_options() -> None:
    """Test StoreConfig with configuration options."""
    from metaxy import InMemoryMetadataStore

    config = StoreConfig(
        type="metaxy.metadata_store.InMemoryMetadataStore",
        config={
            "fallback_stores": ["prod"],
        },
    )

    assert config.type == InMemoryMetadataStore
    assert config.config["fallback_stores"] == ["prod"]


def test_metaxy_config_default() -> None:
    """Test MetaxyConfig with defaults."""
    config = MetaxyConfig()

    assert config.store == "dev"
    assert config.stores == {}


def test_metaxy_config_from_dict() -> None:
    """Test MetaxyConfig from dictionary."""
    config = MetaxyConfig(
        store="staging",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
                config={},
            ),
            "staging": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
                config={"fallback_stores": ["prod"]},
            ),
            "prod": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
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
    """Test loading config from metaxy.toml."""
    from metaxy import InMemoryMetadataStore

    # Create metaxy.toml
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.InMemoryMetadataStore"

[stores.dev.config]
# No config needed for in-memory

[stores.prod]
type = "metaxy.metadata_store.InMemoryMetadataStore"

[stores.prod.config]
fallback_stores = []
""")

    # Load config
    config = MetaxyConfig.load(config_file)

    assert config.store == "dev"
    assert len(config.stores) == 2
    assert config.stores["dev"].type == InMemoryMetadataStore
    assert config.stores["prod"].type == InMemoryMetadataStore


def test_load_from_pyproject_toml(tmp_path: Path) -> None:
    """Test loading config from pyproject.toml [tool.metaxy] section."""
    config_file = tmp_path / "pyproject.toml"
    config_file.write_text("""
[project]
name = "test"

[tool.metaxy]
store = "staging"

[tool.metaxy.stores.staging]
type = "metaxy.metadata_store.InMemoryMetadataStore"

[tool.metaxy.stores.staging.config]
fallback_stores = ["prod"]

[tool.metaxy.stores.prod]
type = "metaxy.metadata_store.InMemoryMetadataStore"

[tool.metaxy.stores.prod.config]
""")

    config = MetaxyConfig.load(config_file)

    assert config.store == "staging"
    assert len(config.stores) == 2
    assert config.stores["staging"].config["fallback_stores"] == ["prod"]


def test_get_store_instantiates_correctly() -> None:
    """Test that get_store properly instantiates a store."""
    config = MetaxyConfig(
        store="dev",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
                config={},
            )
        },
    )

    store = config.get_store("dev")

    assert isinstance(store, InMemoryMetadataStore)
    assert store.fallback_stores == []


def test_get_store_with_fallback_chain() -> None:
    """Test store instantiation with fallback chain."""
    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
                config={"fallback_stores": ["staging"]},
            ),
            "staging": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
                config={"fallback_stores": ["prod"]},
            ),
            "prod": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
                config={},
            ),
        },
    )

    dev_store = config.get_store("dev")

    assert isinstance(dev_store, InMemoryMetadataStore)
    assert len(dev_store.fallback_stores) == 1

    staging_store = dev_store.fallback_stores[0]
    assert isinstance(staging_store, InMemoryMetadataStore)
    assert len(staging_store.fallback_stores) == 1

    prod_store = staging_store.fallback_stores[0]
    assert isinstance(prod_store, InMemoryMetadataStore)
    assert len(prod_store.fallback_stores) == 0


def test_get_store_uses_default() -> None:
    """Test get_store without name uses store."""
    config = MetaxyConfig(
        store="staging",
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
                config={},
            ),
            "staging": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
                config={},
            ),
        },
    )

    # Without name, should use store
    store = config.get_store()
    assert isinstance(store, InMemoryMetadataStore)

    # Verify it's actually staging by checking it has no special config
    # (would need better verification in real scenario)


def test_get_store_nonexistent_raises() -> None:
    """Test get_store raises error for nonexistent store."""
    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
                config={},
            )
        }
    )

    with pytest.raises(ValueError, match="Store 'nonexistent' not found"):
        config.get_store("nonexistent")


def test_config_with_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test environment variable override (pydantic-settings built-in)."""
    from metaxy import InMemoryMetadataStore

    # Create config file
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
store = "dev"

[stores.dev]
type = "metaxy.metadata_store.InMemoryMetadataStore"

[stores.dev.config]
""")

    # Override store via env var
    monkeypatch.setenv("METAXY_STORE", "prod")

    # Also add prod store via env var (pydantic-settings supports this!)
    monkeypatch.setenv(
        "METAXY_STORES__PROD__TYPE", "metaxy.metadata_store.InMemoryMetadataStore"
    )

    config = MetaxyConfig.load(config_file)

    # Env var should override TOML
    assert config.store == "prod"

    # Store from env var should be available
    assert "prod" in config.stores
    assert config.stores["prod"].type == InMemoryMetadataStore


def test_hash_algorithm_must_match_in_fallback_chain() -> None:
    """Test that all stores in a fallback chain must use the same hash algorithm."""
    from metaxy.versioning.types import HashAlgorithm

    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
                config={
                    "hash_algorithm": "sha256",
                    "fallback_stores": ["staging"],
                },
            ),
            "staging": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
                config={"hash_algorithm": "sha256", "fallback_stores": ["prod"]},
            ),
            "prod": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
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
    """Test that hash_algorithm defaults to XXHASH64 when not specified."""
    from metaxy.versioning.types import HashAlgorithm

    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
                config={"fallback_stores": ["prod"]},
            ),
            "prod": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
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
    """Test that conflicting hash_algorithm in fallback store raises error."""
    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
                config={
                    "hash_algorithm": "sha256",
                    "fallback_stores": ["prod"],
                },
            ),
            "prod": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
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
    """Test that instantiated store uses the configured hash algorithm."""
    from metaxy.versioning.types import HashAlgorithm

    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.InMemoryMetadataStore",
                config={"hash_algorithm": "md5"},
            ),
        },
    )

    store = config.get_store("dev")

    # Store should use the configured algorithm
    assert store.hash_algorithm == HashAlgorithm.MD5
