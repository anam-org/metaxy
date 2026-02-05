"""Tests for AccessMode functionality in metadata stores."""

import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from metaxy.ext.metadata_stores.delta import DeltaMetadataStore
from metaxy.ext.metadata_stores.duckdb import DuckDBMetadataStore
from metaxy.metadata_store.system import SystemTableStorage


def test_explicit_read_mode(tmp_path: Path) -> None:
    """Test that stores can be opened explicitly in READ mode."""
    db_path = tmp_path / "test.duckdb"

    # First create the database
    store = DuckDBMetadataStore(db_path, auto_create_tables=True)
    with store.open("w"):
        pass  # Just create the DB

    # Open in READ mode explicitly
    store2 = DuckDBMetadataStore(db_path, auto_create_tables=False)
    with store2:
        # Should be open
        assert store2._is_open


def test_explicit_write_mode(tmp_path: Path) -> None:
    """Test that stores can be opened explicitly in WRITE mode."""
    db_path = tmp_path / "test.duckdb"
    store = DuckDBMetadataStore(db_path, auto_create_tables=True)

    # Open in WRITE mode explicitly
    with store.open("w"):
        assert store._is_open


def test_write_in_write_mode(tmp_path: Path, test_graph, test_features: dict[str, Any]) -> None:
    """Test that write works when opened in WRITE mode."""
    db_path = tmp_path / "test.duckdb"
    store = DuckDBMetadataStore(db_path, auto_create_tables=True)

    # Open in WRITE mode and write
    with store.open("w"):
        metadata = pl.DataFrame(
            {
                "sample_uid": ["s1"],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write(test_features["UpstreamFeatureA"], metadata)

    # Verify data was written
    with store:
        df = store._read_feature(test_features["UpstreamFeatureA"])
        assert df is not None
        result = df.collect().to_polars()
        assert result.height == 1


def test_read_in_read_mode(tmp_path: Path, test_graph, test_features: dict[str, Any]) -> None:
    """Test that read operations work in READ mode."""
    db_path = tmp_path / "test.duckdb"

    # First, create the table and write data in WRITE mode
    with DuckDBMetadataStore(db_path, auto_create_tables=True).open("w") as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": ["s1"],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write(test_features["UpstreamFeatureA"], metadata)

    # Now open in READ mode and read
    store = DuckDBMetadataStore(db_path, auto_create_tables=False)
    with store:
        df = store._read_feature(test_features["UpstreamFeatureA"])
        assert df is not None


def _read_from_store(db_path: Path, result_queue: Any) -> None:
    """Helper function to read from store in a separate process."""
    try:
        store = DuckDBMetadataStore(db_path)
        with store:
            # Try to list tables
            tables = store.conn.list_tables()
            result_queue.put(("success", len(tables)))
    except Exception as e:
        result_queue.put(("error", str(e)))


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="DuckDB concurrent read access only works on Linux (uses range locks); macOS/Windows use exclusive file locks",
)
def test_concurrent_read_access_duckdb(tmp_path: Path, test_graph, test_features: dict[str, Any]) -> None:
    """Test that multiple processes can read concurrently in READ mode."""
    db_path = tmp_path / "test.duckdb"

    # First, create the database and write some data
    with DuckDBMetadataStore(db_path, auto_create_tables=True).open("w") as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": ["s1"],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write(test_features["UpstreamFeatureA"], metadata)

    # Now try to read from multiple processes concurrently
    queue = multiprocessing.Queue()

    # Start first reader
    p1 = multiprocessing.Process(target=_read_from_store, args=(db_path, queue))
    p1.start()

    # Give it time to acquire the lock
    time.sleep(0.1)

    # Start second reader (should succeed because both are READ mode)
    p2 = multiprocessing.Process(target=_read_from_store, args=(db_path, queue))
    p2.start()

    # Wait for both to complete
    p1.join(timeout=5)
    p2.join(timeout=5)

    # Check results
    results = []
    while not queue.empty():
        results.append(queue.get())

    # Both should succeed
    assert len(results) == 2
    for status, value in results:
        assert status == "success", f"Reader failed with: {value}"


def _write_to_store(db_path: Path, sample_id: str, result_queue: Any) -> None:
    """Helper function to write to store in a separate process."""
    try:
        # Import here to avoid pickling issues
        from metaxy.models.feature import BaseFeature, FeatureGraph, FeatureSpec
        from metaxy.models.field import FieldSpec
        from metaxy.models.types import FeatureKey

        # Create test graph
        graph = FeatureGraph()
        with graph.use():

            class UpstreamFeatureA(
                BaseFeature,
                spec=FeatureSpec(
                    key=FeatureKey(["test_stores", "upstream_a"]),
                    id_columns=["sample_uid"],
                    fields=[
                        FieldSpec(key="frames", deps=[]),
                        FieldSpec(key="audio", deps=[]),
                    ],
                ),
            ):
                pass

        store = DuckDBMetadataStore(db_path, auto_create_tables=True)
        with store.open("w"):
            metadata = pl.DataFrame(
                {
                    "sample_uid": [sample_id],
                    "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
                }
            )
            with graph.use():
                store.write(UpstreamFeatureA, metadata)
            result_queue.put(("success", sample_id))
    except Exception as e:
        result_queue.put(("error", str(e)))


def test_write_mode_exclusive_lock_duckdb(tmp_path: Path, test_graph, test_features: dict[str, Any]) -> None:
    """Test that WRITE mode prevents concurrent writes (exclusive lock).

    Note: This is a simplified test that just verifies WRITE mode opens successfully.
    Full concurrent locking behavior is complex to test with multiprocessing and DuckDB.
    """
    db_path = tmp_path / "test.duckdb"

    # Create the database first
    with DuckDBMetadataStore(db_path, auto_create_tables=True).open("w") as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": ["s0"],
                "metaxy_provenance_by_field": [{"frames": "h0", "audio": "h0"}],
            }
        )
        store.write(test_features["UpstreamFeatureA"], metadata)

    # Open in WRITE mode explicitly
    with DuckDBMetadataStore(db_path, auto_create_tables=True).open("w") as store:
        # Should be able to write
        metadata2 = pl.DataFrame(
            {
                "sample_uid": ["s1"],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write(test_features["UpstreamFeatureA"], metadata2)


def test_mode_parameter_passed_to_open(tmp_path: Path) -> None:
    """Test that mode parameter is properly passed to open() method."""
    db_path = tmp_path / "test.duckdb"

    # First create the database
    with DuckDBMetadataStore(db_path, auto_create_tables=True).open("w"):
        pass  # Just create the DB

    # Open with READ mode (default)
    store = DuckDBMetadataStore(db_path, auto_create_tables=False)
    with store:
        # Check that read_only flag is set in connection params (READ mode is default)
        assert store.connection_params.get("read_only") is True

    # Open with WRITE mode
    store2 = DuckDBMetadataStore(db_path, auto_create_tables=False)
    with store2.open("w"):
        # Check that read_only flag is not set (WRITE mode)
        # Note: the flag may be removed by open() or set to False
        assert "read_only" not in store2.connection_params or store2.connection_params.get("read_only") is False


def test_delta_store_modes(test_graph, test_features: dict[str, Any], tmp_path: Path) -> None:
    """Test that DeltaMetadataStore works with explicit modes."""
    store_read = DeltaMetadataStore(root_path=tmp_path / "delta_read")
    store_write = DeltaMetadataStore(root_path=tmp_path / "delta_write")

    # Read mode
    with store_read:
        assert store_read._is_open

    # Write mode
    with store_write.open("w"):
        assert store_write._is_open
        metadata = pl.DataFrame(
            {
                "sample_uid": ["s1"],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store_write.write(test_features["UpstreamFeatureA"], metadata)


def test_mode_reset_between_opens(tmp_path: Path) -> None:
    """Test that store can be reopened with different modes."""
    db_path = tmp_path / "test.duckdb"

    # First create the database
    with DuckDBMetadataStore(db_path, auto_create_tables=True).open("w"):
        pass  # Just create the DB

    # Now test reopening
    store = DuckDBMetadataStore(db_path, auto_create_tables=False)

    # First open (READ)
    with store:
        assert store._is_open

    # After exiting, store should not be open
    assert not store._is_open

    # Second open - still READ
    with store:
        assert store._is_open

    # After exiting again
    assert not store._is_open


def test_record_feature_graph_snapshot_uses_write_mode(tmp_path: Path, test_graph) -> None:
    """Test that push_graph_snapshot works in WRITE mode."""
    db_path = tmp_path / "test.duckdb"

    # Open with WRITE mode
    store = DuckDBMetadataStore(db_path, auto_create_tables=True)

    with store.open("w"):
        result = SystemTableStorage(store).push_graph_snapshot()
        assert result.snapshot_version is not None


def test_drop_feature_metadata_in_write_mode(tmp_path: Path, test_graph, test_features: dict[str, Any]) -> None:
    """Test that drop_feature_metadata works in WRITE mode."""
    db_path = tmp_path / "test.duckdb"

    # Create some data first
    with DuckDBMetadataStore(db_path, auto_create_tables=True).open("w") as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": ["s1"],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write(test_features["UpstreamFeatureA"], metadata)

    # Now open in WRITE mode to drop
    store = DuckDBMetadataStore(db_path, auto_create_tables=False)
    with store.open("w"):
        # Dropping should work
        store.drop_feature_metadata(test_features["UpstreamFeatureA"])


def test_with_store_pattern_works(tmp_path: Path) -> None:
    """Test that 'with store:' pattern works by calling open() internally."""
    db_path = tmp_path / "test.duckdb"

    # Test with auto_create_tables=True (opens in WRITE mode)
    store = DuckDBMetadataStore(db_path, auto_create_tables=True)
    with store:
        assert store._is_open
    assert not store._is_open

    # Test with auto_create_tables=False (opens in READ mode)
    store2 = DuckDBMetadataStore(db_path, auto_create_tables=False)
    with store2:
        assert store2._is_open
    assert not store2._is_open
