"""Store-agnostic tests for persistent metadata stores.

These tests run against all persistent store implementations
(InMemoryMetadataStore, DuckDBMetadataStore, etc.) using pytest-cases parametrization.
"""

from typing import Any

import narwhals as nw
import polars as pl
import pytest

from metaxy._utils import collect_to_polars
from metaxy.metadata_store import (
    FeatureNotFoundError,
    MetadataSchemaError,
    StoreNotOpenError,
)

# Context Manager Tests


def test_store_requires_context_manager(
    persistent_store, test_graph, test_features: dict[str, Any]
) -> None:
    """Test that operations outside context manager raise error."""
    data = pl.DataFrame(
        {
            "sample_id": [1, 2],
            "data_version": [
                {"frames": "h1", "audio": "h1"},
                {"frames": "h2", "audio": "h2"},
            ],
        }
    )

    # Should not be able to write without opening
    with pytest.raises(StoreNotOpenError):
        persistent_store.write_metadata(test_features["UpstreamFeatureA"], data)

    # Should not be able to read without opening
    with pytest.raises(StoreNotOpenError):
        persistent_store.read_metadata(test_features["UpstreamFeatureA"])


def test_store_context_manager(
    persistent_store, test_graph, test_features: dict[str, Any]
) -> None:
    """Test that store works as a context manager."""
    with persistent_store as store:
        # Store should be open
        assert store._is_open

        # Can perform operations
        data = pl.DataFrame(
            {
                "sample_id": [1],
                "data_version": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write_metadata(test_features["UpstreamFeatureA"], data)

    # After exiting, should be closed
    assert not persistent_store._is_open


# Basic CRUD Tests


def test_write_and_read_metadata(
    persistent_store, test_graph, test_features: dict[str, Any]
) -> None:
    """Test basic write and read operations."""
    with persistent_store as store:
        metadata = pl.DataFrame(
            {
                "sample_id": [1, 2, 3],
                "path": ["/data/1.mp4", "/data/2.mp4", "/data/3.mp4"],
                "data_version": [
                    {"frames": "hash1", "audio": "hash1"},
                    {"frames": "hash2", "audio": "hash2"},
                    {"frames": "hash3", "audio": "hash3"},
                ],
            }
        )

        store.write_metadata(test_features["UpstreamFeatureA"], metadata)
        result = collect_to_polars(
            store.read_metadata(test_features["UpstreamFeatureA"])
        )

        assert len(result) == 3
        assert "sample_id" in result.columns
        assert "data_version" in result.columns
        assert "path" in result.columns


def test_write_invalid_schema(
    persistent_store, test_graph, test_features: dict[str, Any]
) -> None:
    """Test that writing without data_version column raises error."""
    with persistent_store as store:
        invalid_df = pl.DataFrame(
            {
                "sample_id": [1, 2, 3],
                "path": ["/a", "/b", "/c"],
            }
        )

        with pytest.raises(MetadataSchemaError, match="data_version"):
            store.write_metadata(test_features["UpstreamFeatureA"], invalid_df)


def test_write_append(
    persistent_store, test_graph, test_features: dict[str, Any]
) -> None:
    """Test that writes are append-only."""
    with persistent_store as store:
        df1 = pl.DataFrame(
            {
                "sample_id": [1, 2],
                "data_version": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                ],
            }
        )

        df2 = pl.DataFrame(
            {
                "sample_id": [3, 4],
                "data_version": [
                    {"frames": "h3", "audio": "h3"},
                    {"frames": "h4", "audio": "h4"},
                ],
            }
        )

        store.write_metadata(test_features["UpstreamFeatureA"], df1)
        store.write_metadata(test_features["UpstreamFeatureA"], df2)

        result = collect_to_polars(
            store.read_metadata(test_features["UpstreamFeatureA"])
        )
        assert len(result) == 4
        assert set(result["sample_id"].to_list()) == {1, 2, 3, 4}


def test_read_with_filters(
    persistent_store, test_graph, test_features: dict[str, Any]
) -> None:
    """Test reading with Polars filter expressions."""
    with persistent_store as store:
        metadata = pl.DataFrame(
            {
                "sample_id": [1, 2, 3],
                "data_version": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                    {"frames": "h3", "audio": "h3"},
                ],
            }
        )
        store.write_metadata(test_features["UpstreamFeatureA"], metadata)

        result = collect_to_polars(
            store.read_metadata(
                test_features["UpstreamFeatureA"], filters=[nw.col("sample_id") > 1]
            )
        )

        assert len(result) == 2
        assert set(result["sample_id"].to_list()) == {2, 3}


def test_read_with_column_selection(
    persistent_store, test_graph, test_features: dict[str, Any]
) -> None:
    """Test reading specific columns."""
    with persistent_store as store:
        metadata = pl.DataFrame(
            {
                "sample_id": [1, 2, 3],
                "path": ["/a", "/b", "/c"],
                "data_version": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                    {"frames": "h3", "audio": "h3"},
                ],
            }
        )
        store.write_metadata(test_features["UpstreamFeatureA"], metadata)

        result = collect_to_polars(
            store.read_metadata(
                test_features["UpstreamFeatureA"], columns=["sample_id", "data_version"]
            )
        )

        assert set(result.columns) == {"sample_id", "data_version"}
        assert "path" not in result.columns


def test_read_nonexistent_feature(
    persistent_store, test_graph, test_features: dict[str, Any]
) -> None:
    """Test that reading nonexistent feature raises error."""
    with persistent_store as store:
        with pytest.raises(FeatureNotFoundError):
            store.read_metadata(test_features["UpstreamFeatureA"])


# Feature Existence Tests


def test_has_feature_local(
    persistent_store, test_graph, test_features: dict[str, Any]
) -> None:
    """Test has_feature for local store."""
    with persistent_store as store:
        assert not store.has_feature(
            test_features["UpstreamFeatureA"], check_fallback=False
        )

        metadata = pl.DataFrame(
            {
                "sample_id": [1],
                "data_version": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write_metadata(test_features["UpstreamFeatureA"], metadata)

        assert store.has_feature(
            test_features["UpstreamFeatureA"], check_fallback=False
        )
        assert not store.has_feature(
            test_features["UpstreamFeatureB"], check_fallback=False
        )


def test_list_features(
    persistent_store, test_graph, test_features: dict[str, Any]
) -> None:
    """Test listing features.

    Args:
        persistent_store: Store fixture (unopened)
        test_graph: Registry with test features
    """
    with persistent_store as store:
        # Empty initially
        features = store.list_features()
        assert len(features) == 0

        # Add features
        data_a = pl.DataFrame(
            {
                "sample_id": [1],
                "data_version": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write_metadata(test_features["UpstreamFeatureA"], data_a)

        data_b = pl.DataFrame(
            {
                "sample_id": [1],
                "data_version": [{"default": "h1"}],
            }
        )
        store.write_metadata(test_features["UpstreamFeatureB"], data_b)

        features = store.list_features()
        assert len(features) == 2
        feature_strs = {f.to_string() for f in features}
        assert feature_strs == {"test_stores/upstream_a", "test_stores/upstream_b"}


# Data Version Calculation Tests
# NOTE: The calculate_and_write_data_versions API was removed.
# Data version calculation is now handled differently in the new architecture.
# These tests have been removed as they tested a deprecated API.


# System Tables Tests


def test_system_tables(
    persistent_store, test_graph, test_features: dict[str, Any]
) -> None:
    """Test that system tables work correctly.

    Args:
        persistent_store: Store fixture (unopened)
        test_graph: Registry with test features
    """
    from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

    graph, _ = test_graph

    with graph.use(), persistent_store as store:
        # Write data and record version
        data = pl.DataFrame(
            {
                "sample_id": [1, 2],
                "data_version": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                ],
            }
        )
        store.write_metadata(test_features["UpstreamFeatureA"], data)
        store.record_feature_graph_snapshot()

        # Read system table
        version_history = collect_to_polars(
            store.read_metadata(FEATURE_VERSIONS_KEY, current_only=False)
        )

        assert len(version_history) > 0
        assert "feature_key" in version_history.columns
        assert "feature_version" in version_history.columns
        assert "recorded_at" in version_history.columns


# Display/Repr Tests


def test_display(persistent_store) -> None:
    """Test display method works when store is closed and open."""
    # Should work when closed
    display_closed = persistent_store.display()
    assert len(display_closed) > 0

    with persistent_store as store:
        # Should work when open
        display_open = store.display()
        assert len(display_open) > 0


def test_repr(persistent_store) -> None:
    """Test /repr/ method."""
    with persistent_store as store:
        repr_str = repr(store)
        assert len(repr_str) > 0


# Nested Context Manager Tests


def test_nested_context_managers(
    persistent_store, test_graph, test_features: dict[str, Any]
) -> None:
    """Test that nested context managers work correctly.

    Args:
        persistent_store: Store fixture (unopened)
        test_graph: Registry with test features
    """
    with persistent_store as store1:
        # Nest another context (simulates fallback opening)
        with store1 as store2:
            assert store1 is store2
            assert store1._is_open

            metadata = pl.DataFrame(
                {
                    "sample_id": [1],
                    "data_version": [{"frames": "h1", "audio": "h1"}],
                }
            )
            store1.write_metadata(test_features["UpstreamFeatureA"], metadata)

        # Should still be open after inner context exits
        assert store1._is_open
        result = collect_to_polars(
            store1.read_metadata(test_features["UpstreamFeatureA"])
        )
        assert len(result) == 1

    # Now should be closed after outer context exits
    assert not persistent_store._is_open


# Multiple Features Tests


def test_multiple_features(
    persistent_store, test_graph, test_features: dict[str, Any]
) -> None:
    """Test storing multiple features in same store.

    Args:
        persistent_store: Store fixture (unopened)
        test_graph: Registry with test features
    """
    with persistent_store as store:
        # Write feature A
        data_a = pl.DataFrame(
            {
                "sample_id": [1, 2],
                "data_version": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                ],
            }
        )
        store.write_metadata(test_features["UpstreamFeatureA"], data_a)

        # Write feature B
        data_b = pl.DataFrame(
            {
                "sample_id": [1, 2, 3],
                "data_version": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
            }
        )
        store.write_metadata(test_features["UpstreamFeatureB"], data_b)

        # Read both
        result_a = collect_to_polars(
            store.read_metadata(test_features["UpstreamFeatureA"])
        )
        result_b = collect_to_polars(
            store.read_metadata(test_features["UpstreamFeatureB"])
        )

        assert len(result_a) == 2
        assert len(result_b) == 3
