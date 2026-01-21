"""Tests for ADBC federation support via DuckDB adbc_scanner extension.

Note:
    These tests verify the API exists but don't test actual federation
    since that would require external databases. Full integration tests
    for federation should be added separately.
"""

import pytest

from metaxy import HashAlgorithm
from metaxy.metadata_store.duckdb import DuckDBMetadataStore


def test_duckdb_has_adbc_scanner_methods(tmp_path):
    """Test that DuckDB store has ADBC scanner methods."""
    store = DuckDBMetadataStore(
        database=tmp_path / "test.duckdb",
        hash_algorithm=HashAlgorithm.XXHASH64,
    )

    # Verify methods exist
    assert hasattr(store, "install_adbc_scanner")
    assert hasattr(store, "adbc_connect")
    assert hasattr(store, "adbc_disconnect")
    assert hasattr(store, "adbc_scan")
    assert hasattr(store, "adbc_scan_table")

    # Verify they're callable
    assert callable(store.install_adbc_scanner)
    assert callable(store.adbc_connect)
    assert callable(store.adbc_disconnect)
    assert callable(store.adbc_scan)
    assert callable(store.adbc_scan_table)


def test_adbc_scanner_requires_open_store(tmp_path):
    """Test that ADBC scanner methods require an open store."""
    store = DuckDBMetadataStore(
        database=tmp_path / "test.duckdb",
        hash_algorithm=HashAlgorithm.XXHASH64,
    )

    # Should raise when store is not open
    with pytest.raises(RuntimeError, match="connection is not open"):
        store.install_adbc_scanner()


@pytest.mark.slow
def test_adbc_scanner_extension_installation(tmp_path):
    """Test that adbc_scanner extension can be installed (if available).

    This test is marked as slow because it downloads the extension.
    It may fail in environments where the extension is not available.
    """
    store = DuckDBMetadataStore(
        database=tmp_path / "test.duckdb",
        hash_algorithm=HashAlgorithm.XXHASH64,
    )

    with store:
        try:
            # Try to install extension
            store.install_adbc_scanner()
            # If successful, extension is available
            print("✓ adbc_scanner extension installed successfully")
        except Exception as e:
            # Extension may not be available in all environments
            pytest.skip(f"adbc_scanner extension not available: {e}")
