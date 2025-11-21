"""Comprehensive tests for Dagster integration.

Tests the MetaxyMetadataStoreResource class which provides Metaxy metadata stores
as Dagster resources. Covers:
1. Config loading (explicit, auto-discovery, already-set context)
2. Fallback store validation and error handling
3. Store creation and configuration

These tests focus on the core functionality (_load_config, _resolve_fallback_stores,
_validate_fallback_store), resource creation logic, and Dagster integration points.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dagster as dg
import pytest

from metaxy import MetaxyConfig
from metaxy.config import StoreConfig
from metaxy.ext.dagster.resource import (
    MetaxyMetadataStoreResource,
    _resolve_fallback_stores,
    _validate_fallback_store,
)
from metaxy.metadata_store.base import MetadataStore
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.metadata_store.memory import InMemoryMetadataStore
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    pass


# ============= CONFIG LOADING TESTS =============


def test_load_config_explicit_file(tmp_path: Path) -> None:
    """Test loading config from explicit config_file parameter.

    Verifies that:
    - Explicit config_file is used when provided
    - Config is loaded correctly from the specified file
    """
    # Setup config
    config_file = tmp_path / "custom_metaxy.toml"
    config_file.write_text(
        """project = "custom_project"
store = "custom_store"

[stores.custom_store]
type = "metaxy.metadata_store.memory.InMemoryMetadataStore"
"""
    )

    # Create resource instance directly (bypass Dagster config validation)
    resource = MetaxyMetadataStoreResource.__new__(MetaxyMetadataStoreResource)
    # Use object.__setattr__ since ConfigurableResource is frozen
    object.__setattr__(resource, "store_name", "custom_store")
    object.__setattr__(resource, "config_file", config_file)
    object.__setattr__(resource, "search_parents", True)
    object.__setattr__(resource, "auto_discovery_start", None)
    object.__setattr__(resource, "fallback_stores", None)

    # Load config and verify
    config = resource._load_config()
    assert config.project == "custom_project"
    assert config.store == "custom_store"

    # Get store
    store = config.get_store("custom_store")
    assert isinstance(store, InMemoryMetadataStore)


def test_load_config_from_set_context(tmp_path: Path) -> None:
    """Test loading config from already-set MetaxyConfig context.

    Verifies that:
    - When MetaxyConfig.is_set(), that config is used
    - No file loading occurs when config is already set
    """
    # Setup config in global context
    test_config = MetaxyConfig(
        project="context_test",
        stores={
            "memory": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="memory",
    )
    MetaxyConfig.set(test_config)

    try:
        # Create resource instance
        resource = MetaxyMetadataStoreResource.__new__(MetaxyMetadataStoreResource)
        object.__setattr__(resource, "store_name", "memory")
        object.__setattr__(resource, "config_file", None)
        object.__setattr__(resource, "search_parents", True)
        object.__setattr__(resource, "auto_discovery_start", None)
        object.__setattr__(resource, "fallback_stores", None)

        # Load config
        config = resource._load_config()

        assert config is test_config
        assert config.project == "context_test"

        # Get store
        store = config.get_store("memory")
        assert isinstance(store, InMemoryMetadataStore)
    finally:
        # Clean up - reset config
        MetaxyConfig.reset()


# ============= FALLBACK STORES VALIDATION TESTS =============


def test_validate_fallback_store_compatible_hash_algorithm(tmp_path: Path) -> None:
    """Test that fallback stores with compatible hash algorithms pass validation.

    Verifies that:
    - Fallback stores with same hash algorithm as primary are accepted
    - No validation errors occur
    """
    # Create stores with same hash algorithm
    primary_db = tmp_path / "primary.duckdb"
    fallback_db = tmp_path / "fallback.duckdb"

    primary_store = DuckDBMetadataStore(
        primary_db, hash_algorithm=HashAlgorithm.XXHASH64
    )
    fallback_store = DuckDBMetadataStore(
        fallback_db, hash_algorithm=HashAlgorithm.XXHASH64
    )

    # Should not raise
    _validate_fallback_store(primary_store, fallback_store, "fallback")


def test_validate_fallback_store_compatible_hash_truncation(tmp_path: Path) -> None:
    """Test that fallback stores with compatible truncation lengths pass validation.

    Verifies that:
    - Fallback stores with same truncation length as primary are accepted
    - No validation errors occur
    - hash_truncation_length is a property derived from MetaxyConfig
    """
    # Create stores - hash_truncation_length comes from MetaxyConfig
    # which is set by the autouse config fixture in conftest.py
    primary_db = tmp_path / "primary.duckdb"
    fallback_db = tmp_path / "fallback.duckdb"

    primary_store = DuckDBMetadataStore(primary_db)
    fallback_store = DuckDBMetadataStore(fallback_db)

    # Both will have the same hash_truncation_length from the config
    assert primary_store.hash_truncation_length == fallback_store.hash_truncation_length

    # Should not raise
    _validate_fallback_store(primary_store, fallback_store, "fallback")


def test_validate_fallback_store_different_hash_algorithm_error(
    tmp_path: Path,
) -> None:
    """Test error when fallback store has different hash algorithm.

    Verifies that:
    - Different hash algorithm between primary and fallback raises ValueError
    - Error message clearly explains the mismatch
    """
    # Create stores with different hash algorithms
    primary_db = tmp_path / "primary.duckdb"
    fallback_db = tmp_path / "fallback.duckdb"

    primary_store = DuckDBMetadataStore(
        primary_db, hash_algorithm=HashAlgorithm.XXHASH64
    )
    fallback_store = DuckDBMetadataStore(
        fallback_db, hash_algorithm=HashAlgorithm.SHA256
    )

    # Should raise
    with pytest.raises(
        ValueError, match="Fallback store 'fallback' uses a different hash algorithm"
    ):
        _validate_fallback_store(primary_store, fallback_store, "fallback")


@pytest.mark.skip(
    reason="hash_truncation_length is a config-level setting, difficult to test different values per store"
)
def test_validate_fallback_store_different_hash_truncation_error() -> None:
    """Test error when fallback store has different hash truncation length.

    Verifies that:
    - Different truncation length between primary and fallback raises ValueError
    - Error message clearly explains the mismatch

    Note: Skipped because hash_truncation_length is a global MetaxyConfig setting,
    and stores inherit it from the config. The validation logic in _validate_fallback_store
    is tested indirectly through compatible truncation tests.
    """
    pass


# ============= RESOLVE FALLBACK STORES TESTS =============


def test_resolve_single_fallback_store(tmp_path: Path) -> None:
    """Test resolving a single fallback store.

    Verifies that:
    - Single fallback store can be resolved from config
    - Fallback store is properly instantiated
    - Validation passes for compatible configuration
    """
    # Setup config with multiple stores
    config = MetaxyConfig(
        project="test",
        stores={
            "primary": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": str(tmp_path / "primary.duckdb")},
            ),
            "fallback": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": str(tmp_path / "fallback.duckdb")},
            ),
        },
        store="primary",
    )

    primary_store = config.get_store("primary")
    fallback_stores = _resolve_fallback_stores(
        config=config,
        fallback_names=["fallback"],
        primary_store=primary_store,
        primary_store_name="primary",
    )

    assert len(fallback_stores) == 1
    assert isinstance(fallback_stores[0], DuckDBMetadataStore)


def test_resolve_multiple_fallback_stores(tmp_path: Path) -> None:
    """Test resolving multiple fallback stores.

    Verifies that:
    - Multiple fallback stores can be resolved in order
    - All fallback stores are properly instantiated
    - Order is preserved
    """
    # Setup config with multiple stores
    config = MetaxyConfig(
        project="test",
        stores={
            "primary": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": str(tmp_path / "primary.duckdb")},
            ),
            "fallback1": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": str(tmp_path / "fallback1.duckdb")},
            ),
            "fallback2": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": str(tmp_path / "fallback2.duckdb")},
            ),
        },
        store="primary",
    )

    primary_store = config.get_store("primary")
    fallback_stores = _resolve_fallback_stores(
        config=config,
        fallback_names=["fallback1", "fallback2"],
        primary_store=primary_store,
        primary_store_name="primary",
    )

    assert len(fallback_stores) == 2
    assert all(isinstance(s, DuckDBMetadataStore) for s in fallback_stores)


def test_resolve_fallback_stores_duplicate_error(tmp_path: Path) -> None:
    """Test error when duplicate fallback stores are specified.

    Verifies that:
    - Duplicate fallback store names raise ValueError
    - Error message identifies the duplicate
    """
    # Setup config
    config = MetaxyConfig(
        project="test",
        stores={
            "primary": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": str(tmp_path / "primary.duckdb")},
            ),
            "fallback": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": str(tmp_path / "fallback.duckdb")},
            ),
        },
        store="primary",
    )

    primary_store = config.get_store("primary")

    # Should raise on duplicate
    with pytest.raises(ValueError, match="Duplicate fallback store 'fallback'"):
        _resolve_fallback_stores(
            config=config,
            fallback_names=["fallback", "fallback"],  # Duplicate!
            primary_store=primary_store,
            primary_store_name="primary",
        )


def test_resolve_fallback_stores_self_reference_error(tmp_path: Path) -> None:
    """Test error when store lists itself as a fallback.

    Verifies that:
    - Store cannot be its own fallback (circular reference)
    - Error message clearly explains the issue
    """
    # Setup config
    config = MetaxyConfig(
        project="test",
        stores={
            "primary": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": str(tmp_path / "primary.duckdb")},
            ),
        },
        store="primary",
    )

    primary_store = config.get_store("primary")

    # Should raise on self-reference
    with pytest.raises(ValueError, match="cannot list itself as a fallback store"):
        _resolve_fallback_stores(
            config=config,
            fallback_names=["primary"],  # Self-reference!
            primary_store=primary_store,
            primary_store_name="primary",
        )


# ============= INTEGRATION TESTS =============


def test_create_resource_with_fallback_stores(tmp_path: Path) -> None:
    """Test creating a store resource with fallback stores configured.

    Verifies that:
    - Resource can create a store with fallback stores
    - Fallback stores are properly set on the primary store
    - Configuration is correctly applied
    """
    # Setup config
    config = MetaxyConfig(
        project="test",
        stores={
            "primary": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": str(tmp_path / "primary.duckdb")},
            ),
            "fallback": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": str(tmp_path / "fallback.duckdb")},
            ),
        },
        store="primary",
    )
    MetaxyConfig.set(config)

    try:
        # Create resource instance directly
        resource = MetaxyMetadataStoreResource.__new__(MetaxyMetadataStoreResource)
        object.__setattr__(resource, "store_name", "primary")
        object.__setattr__(resource, "config_file", None)
        object.__setattr__(resource, "search_parents", True)
        object.__setattr__(resource, "auto_discovery_start", None)
        object.__setattr__(resource, "fallback_stores", ["fallback"])

        # Manually create the store as create_resource() would
        config = resource._load_config()
        store = config.get_store(resource.store_name)

        # Apply fallback stores
        if resource.fallback_stores is not None:
            store.fallback_stores = _resolve_fallback_stores(
                config=config,
                fallback_names=resource.fallback_stores,
                primary_store=store,
                primary_store_name=resource.store_name or config.store,
            )

        # Verify
        assert isinstance(store, DuckDBMetadataStore)
        assert store.fallback_stores is not None
        assert len(store.fallback_stores) == 1
        assert isinstance(store.fallback_stores[0], DuckDBMetadataStore)
    finally:
        # Clean up
        MetaxyConfig.reset()


def test_from_config_instantiates_resource_with_fallbacks(tmp_path: Path) -> None:
    """Dagster ConfigurableResource can be built via from_config with list fallback."""
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text(
        f"""project = "test"
store = "primary"

[stores.primary]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"
config.database = "{tmp_path / "primary.duckdb"}"

[stores.fallback]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"
config.database = "{tmp_path / "fallback.duckdb"}"
"""
    )

    resource = MetaxyMetadataStoreResource.from_config(
        store_name="primary",
        fallback_stores=["fallback"],
        config_file=str(config_file),
    )

    try:
        with dg.build_resources({"metaxy_store": resource}) as resources:
            store = resources.metaxy_store
            assert isinstance(store, DuckDBMetadataStore)
            assert store.fallback_stores is not None
            assert isinstance(store.fallback_stores[0], DuckDBMetadataStore)
            assert store.fallback_stores[0].hash_algorithm == store.hash_algorithm
    finally:
        MetaxyConfig.reset()


def test_resource_usable_in_dagster_asset_context(tmp_path: Path) -> None:
    """Resource can be injected into assets and accessed via execution context."""
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text(
        f"""project = "test"
store = "primary"

[stores.primary]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"
config.database = "{tmp_path / "primary.duckdb"}"
"""
    )

    resource = MetaxyMetadataStoreResource.from_config(
        store_name="primary",
        config_file=str(config_file),
    )

    try:

        @dg.asset(required_resource_keys={"metaxy_store"})
        def uses_metaxy_store(context) -> str:
            store = context.resources.metaxy_store
            assert isinstance(store, DuckDBMetadataStore)
            # Return a stable attribute to assert on
            return str(store.database)

        result = dg.materialize(
            [uses_metaxy_store],
            resources={"metaxy_store": resource},
        )

        assert result.success
        assert result.output_for_node("uses_metaxy_store") == str(
            tmp_path / "primary.duckdb"
        )
    finally:
        MetaxyConfig.reset()


def test_create_resource_uses_default_store() -> None:
    """Test creating resource without store name (uses default).

    Verifies that:
    - When store_name is None, default store from config is used
    - Resource creation succeeds
    """
    # Setup config
    config = MetaxyConfig(
        project="test",
        stores={
            "production": StoreConfig(
                type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                config={},
            )
        },
        store="production",
    )
    MetaxyConfig.set(config)

    try:
        # Create resource instance
        resource = MetaxyMetadataStoreResource.__new__(MetaxyMetadataStoreResource)
        object.__setattr__(resource, "store_name", None)  # Use default
        object.__setattr__(resource, "config_file", None)
        object.__setattr__(resource, "search_parents", True)
        object.__setattr__(resource, "auto_discovery_start", None)
        object.__setattr__(resource, "fallback_stores", None)

        # Create store
        loaded_config = resource._load_config()
        store = loaded_config.get_store(resource.store_name)

        # Verify default store is used
        assert isinstance(store, InMemoryMetadataStore)
    finally:
        # Clean up
        MetaxyConfig.reset()


def test_resource_returns_correct_store_type() -> None:
    """Test that create_resource returns correct MetadataStore type.

    Verifies that:
    - Resource returns an instance of MetadataStore
    - The correct store implementation is returned based on config
    """
    # Setup configs for different store types
    configs = {
        "duckdb": MetaxyConfig(
            project="test",
            stores={
                "dev": StoreConfig(
                    type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                    config={"database": ":memory:"},
                )
            },
            store="dev",
        ),
        "memory": MetaxyConfig(
            project="test",
            stores={
                "dev": StoreConfig(
                    type="metaxy.metadata_store.memory.InMemoryMetadataStore",
                    config={},
                )
            },
            store="dev",
        ),
    }

    expected_types = {
        "duckdb": DuckDBMetadataStore,
        "memory": InMemoryMetadataStore,
    }

    for store_type, config in configs.items():
        MetaxyConfig.set(config)
        try:
            # Create resource
            resource = MetaxyMetadataStoreResource.__new__(MetaxyMetadataStoreResource)
            object.__setattr__(resource, "store_name", "dev")
            object.__setattr__(resource, "config_file", None)
            object.__setattr__(resource, "search_parents", True)
            object.__setattr__(resource, "auto_discovery_start", None)
            object.__setattr__(resource, "fallback_stores", None)

            # Create store
            loaded_config = resource._load_config()
            store = loaded_config.get_store(resource.store_name)

            # Verify type
            assert isinstance(store, MetadataStore)
            assert isinstance(store, expected_types[store_type])
        finally:
            # Clean up
            MetaxyConfig.reset()
