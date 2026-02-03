"""Tests for MetadataStore.name attribute and display formatting."""

from pathlib import Path

import pytest
from pytest_cases import parametrize_with_cases

from metaxy import HashAlgorithm
from metaxy.config import MetaxyConfig, StoreConfig
from metaxy.metadata_store import MetadataStore
from metaxy.metadata_store.delta import DeltaMetadataStore
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.metadata_store.lancedb import LanceDBMetadataStore

from .conftest import AllStoresCases


class NamedStoreCases:
    """Store cases with names configured."""

    @pytest.mark.delta
    def case_delta_named(self, tmp_path: Path) -> MetadataStore:
        return DeltaMetadataStore(
            root_path=tmp_path / "delta_store",
            hash_algorithm=HashAlgorithm.XXHASH64,
            name="prod",
        )

    @pytest.mark.duckdb
    def case_duckdb_named(self, tmp_path: Path) -> MetadataStore:
        return DuckDBMetadataStore(
            database=tmp_path / "test.duckdb",
            hash_algorithm=HashAlgorithm.XXHASH64,
            extensions=["hashfuncs"],
            name="staging",
        )

    @pytest.mark.lancedb
    def case_lancedb_named(self, tmp_path: Path) -> MetadataStore:
        return LanceDBMetadataStore(
            uri=tmp_path / "lancedb_store",
            hash_algorithm=HashAlgorithm.XXHASH64,
            name="dev",
        )


@parametrize_with_cases("store", cases=AllStoresCases)
def test_name_is_none_by_default(store: MetadataStore):
    """Test that store.name is None when not explicitly provided."""
    assert store.name is None


@parametrize_with_cases("store", cases=AllStoresCases)
def test_display_without_name(store: MetadataStore):
    """Test that display() works normally without a name."""
    display = store.display()
    # Should not have brackets for name prefix
    assert not display.startswith("[")
    # Should contain the class name
    assert store.__class__.__name__ in display


@parametrize_with_cases("store", cases=NamedStoreCases)
def test_name_property_returns_configured_name(store: MetadataStore):
    """Test that store.name returns the configured name."""
    assert store.name is not None
    assert store.name in ("prod", "staging", "dev")


@parametrize_with_cases("store", cases=NamedStoreCases)
def test_display_without_name_even_when_configured(store: MetadataStore):
    """Test that display() does NOT include the name prefix."""
    display = store.display()
    # Should NOT start with name in brackets
    assert not display.startswith("[")
    # Should contain the class name
    assert store.__class__.__name__ in display


@parametrize_with_cases("store", cases=NamedStoreCases)
def test_repr_includes_name(store: MetadataStore):
    """Test that __repr__ includes the name prefix when configured."""
    repr_str = repr(store)
    # Should start with name in brackets
    assert repr_str.startswith(f"[{store.name}]")
    # Should contain the class name after the prefix
    assert store.__class__.__name__ in repr_str
    # Should include the display() content
    assert store.display() in repr_str


def test_store_from_config_gets_name(tmp_path: Path):
    """Test that stores created via MetaxyConfig.get_store() receive the config key as name."""
    config = MetaxyConfig(
        stores={
            "my_store": StoreConfig(
                type="metaxy.metadata_store.duckdb.DuckDBMetadataStore",
                config={"database": str(tmp_path / "test.duckdb")},
            )
        }
    )

    with config.use():
        store = config.get_store("my_store")

    assert store.name == "my_store"
    # display() should NOT include the name
    assert not store.display().startswith("[")
    # repr() SHOULD include the name
    assert repr(store).startswith("[my_store]")
    assert "DuckDBMetadataStore" in repr(store)
