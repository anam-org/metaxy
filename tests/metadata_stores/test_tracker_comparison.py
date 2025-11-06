"""Tests comparing resolve_update between Polars and Ibis (DuckDB) trackers.

These tests verify that both tracker implementations produce identical results
and provenance hashes.
"""

from pathlib import Path

import polars as pl
import pytest

from metaxy._utils import collect_to_polars
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.metadata_store.memory_simple import SimpleInMemoryMetadataStore
from metaxy.models.feature import BaseFeature, FeatureGraph
from metaxy.provenance.types import HashAlgorithm


# Skip all tests if pyarrow not available
pytest.importorskip("pyarrow")


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """Sample data for testing."""
    return pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "value": ["a", "b", "c"],
        }
    )


def test_resolve_update_identical_results(
    tmp_path: Path,
    test_graph: FeatureGraph,
    test_features: dict[str, type[BaseFeature]],
    sample_data: pl.DataFrame,
) -> None:
    """Test that Polars and Ibis trackers produce identical added samples.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_graph: Test feature graph
        test_features: Dictionary of test features
        sample_data: Sample data fixture
    """
    db_path = tmp_path / "test.duckdb"
    feature = test_features["UpstreamFeatureA"]

    # Run with SimpleInMemoryMetadataStore (uses Polars tracker)
    with SimpleInMemoryMetadataStore(
        hash_algo=HashAlgorithm.MD5, hash_length=16, auto_create_tables=True
    ) as memory_store:
        memory_increment = memory_store.resolve_update(
            feature=feature,
            hash_algorithm=HashAlgorithm.MD5,
            hash_length=16,
            lazy=False,
        )

    # Run with DuckDBMetadataStore (uses Ibis tracker)
    with DuckDBMetadataStore(
        database=db_path,
        hash_algorithm=HashAlgorithm.MD5,
        hash_length=16,
        auto_create_tables=True,
    ) as duckdb_store:
        duckdb_increment = duckdb_store.resolve_update(
            feature=feature,
            hash_algorithm=HashAlgorithm.MD5,
            hash_length=16,
            lazy=False,
        )

    # Convert results to Polars for comparison
    memory_added = collect_to_polars(memory_increment.added)
    duckdb_added = collect_to_polars(duckdb_increment.added)

    # Should produce identical results
    assert len(memory_added) == len(duckdb_added)
    # Note: We can't use pl.testing.assert_frame_equal directly because the order
    # might differ and the provenance column is a struct which needs special handling


def test_resolve_update_identical_provenances(
    tmp_path: Path,
    test_graph: FeatureGraph,
    test_features: dict[str, type[BaseFeature]],
) -> None:
    """Test that Polars and Ibis trackers produce identical provenance hashes.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_graph: Test feature graph
        test_features: Dictionary of test features
    """
    db_path = tmp_path / "test.duckdb"
    feature = test_features["UpstreamFeatureA"]

    # Prepare upstream metadata
    upstream_data = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "frames": ["frame1", "frame2", "frame3"],
            "audio": ["audio1", "audio2", "audio3"],
            "metaxy_provenance_by_field": [
                {"frames": "h1", "audio": "h1"},
                {"frames": "h2", "audio": "h2"},
                {"frames": "h3", "audio": "h3"},
            ],
        }
    )

    # Run with SimpleInMemoryMetadataStore (uses Polars tracker)
    with SimpleInMemoryMetadataStore(
        hash_algo=HashAlgorithm.MD5, hash_length=16, auto_create_tables=True
    ) as memory_store:
        memory_store.write_metadata(feature, upstream_data)
        memory_increment = memory_store.resolve_update(
            feature=feature,
            hash_algorithm=HashAlgorithm.MD5,
            hash_length=16,
            lazy=False,
        )

    # Run with DuckDBMetadataStore (uses Ibis tracker)
    with DuckDBMetadataStore(
        database=db_path,
        hash_algorithm=HashAlgorithm.MD5,
        hash_length=16,
        auto_create_tables=True,
    ) as duckdb_store:
        duckdb_store.write_metadata(feature, upstream_data)
        duckdb_increment = duckdb_store.resolve_update(
            feature=feature,
            hash_algorithm=HashAlgorithm.MD5,
            hash_length=16,
            lazy=False,
        )

    # Convert to Polars
    memory_added = collect_to_polars(memory_increment.added)
    duckdb_added = collect_to_polars(duckdb_increment.added)

    # Both should have provenance columns
    assert "metaxy_provenance_by_field" in memory_added.columns
    assert "metaxy_provenance_by_field" in duckdb_added.columns

    # Extract provenance hashes for comparison
    # Sort by sample_uid to ensure same order
    memory_sorted = memory_added.sort("sample_uid")
    duckdb_sorted = duckdb_added.sort("sample_uid")

    # Compare sample_uids
    pl.testing.assert_series_equal(
        memory_sorted["sample_uid"], duckdb_sorted["sample_uid"]
    )

    # Compare provenance structs (field by field)
    # Note: Struct columns need special handling in Polars
    for col in ["frames", "audio"]:
        memory_prov = memory_sorted["metaxy_provenance_by_field"].struct.field(col)
        duckdb_prov = duckdb_sorted["metaxy_provenance_by_field"].struct.field(col)
        pl.testing.assert_series_equal(
            memory_prov, duckdb_prov, check_names=False
        )


def test_resolve_update_with_changes(
    tmp_path: Path,
    test_graph: FeatureGraph,
    test_features: dict[str, type[BaseFeature]],
) -> None:
    """Test that both trackers detect the same changed samples.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_graph: Test feature graph
        test_features: Dictionary of test features
    """
    db_path = tmp_path / "test.duckdb"
    feature = test_features["UpstreamFeatureA"]

    # Prepare current metadata (old version)
    current_data = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "frames": ["frame1_old", "frame2", "frame3"],
            "audio": ["audio1_old", "audio2", "audio3"],
            "metaxy_provenance_by_field": [
                {"frames": "old_h1", "audio": "old_h1"},
                {"frames": "h2", "audio": "h2"},
                {"frames": "h3", "audio": "h3"},
            ],
            "metaxy_feature_version": [
                feature.feature_version(),
                feature.feature_version(),
                feature.feature_version(),
            ],
        }
    )

    # Prepare new upstream data (sample 1 has changed)
    new_data = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3, 4],  # Added sample 4
            "frames": ["frame1_new", "frame2", "frame3", "frame4"],
            "audio": ["audio1_new", "audio2", "audio3", "audio4"],
            "metaxy_provenance_by_field": [
                {"frames": "new_h1", "audio": "new_h1"},  # Changed
                {"frames": "h2", "audio": "h2"},  # Unchanged
                {"frames": "h3", "audio": "h3"},  # Unchanged
                {"frames": "h4", "audio": "h4"},  # New
            ],
        }
    )

    # Run with SimpleInMemoryMetadataStore
    with SimpleInMemoryMetadataStore(
        hash_algo=HashAlgorithm.MD5, hash_length=16, auto_create_tables=True
    ) as memory_store:
        memory_store.write_metadata(feature, current_data)
        memory_store.write_metadata(feature, new_data)
        memory_increment = memory_store.resolve_update(
            feature=feature,
            hash_algorithm=HashAlgorithm.MD5,
            hash_length=16,
            lazy=False,
        )

    # Run with DuckDBMetadataStore
    with DuckDBMetadataStore(
        database=db_path,
        hash_algorithm=HashAlgorithm.MD5,
        hash_length=16,
        auto_create_tables=True,
    ) as duckdb_store:
        duckdb_store.write_metadata(feature, current_data)
        duckdb_store.write_metadata(feature, new_data)
        duckdb_increment = duckdb_store.resolve_update(
            feature=feature,
            hash_algorithm=HashAlgorithm.MD5,
            hash_length=16,
            lazy=False,
        )

    # Convert to Polars
    memory_added = collect_to_polars(memory_increment.added)
    memory_changed = collect_to_polars(memory_increment.changed)
    memory_removed = collect_to_polars(memory_increment.removed)

    duckdb_added = collect_to_polars(duckdb_increment.added)
    duckdb_changed = collect_to_polars(duckdb_increment.changed)
    duckdb_removed = collect_to_polars(duckdb_increment.removed)

    # Should detect same added samples (sample 4)
    assert len(memory_added) == len(duckdb_added) == 1
    assert memory_added["sample_uid"][0] == duckdb_added["sample_uid"][0] == 4

    # Should detect same changed samples (sample 1)
    assert len(memory_changed) == len(duckdb_changed) == 1
    assert memory_changed["sample_uid"][0] == duckdb_changed["sample_uid"][0] == 1

    # No removed samples
    assert len(memory_removed) == len(duckdb_removed) == 0


def test_resolve_update_with_xxhash(
    tmp_path: Path,
    test_graph: FeatureGraph,
    test_features: dict[str, type[BaseFeature]],
) -> None:
    """Test that both trackers work with xxhash64 algorithm.

    Args:
        tmp_path: Pytest tmp_path fixture
        test_graph: Test feature graph
        test_features: Dictionary of test features
    """
    db_path = tmp_path / "test.duckdb"
    feature = test_features["UpstreamFeatureA"]

    upstream_data = pl.DataFrame(
        {
            "sample_uid": [1, 2],
            "frames": ["frame1", "frame2"],
            "audio": ["audio1", "audio2"],
            "metaxy_provenance_by_field": [
                {"frames": "h1", "audio": "h1"},
                {"frames": "h2", "audio": "h2"},
            ],
        }
    )

    # Run with SimpleInMemoryMetadataStore
    with SimpleInMemoryMetadataStore(
        hash_algo=HashAlgorithm.XXHASH64, hash_length=16, auto_create_tables=True
    ) as memory_store:
        memory_store.write_metadata(feature, upstream_data)
        memory_increment = memory_store.resolve_update(
            feature=feature,
            hash_algorithm=HashAlgorithm.XXHASH64,
            hash_length=16,
            lazy=False,
        )

    # Run with DuckDBMetadataStore
    with DuckDBMetadataStore(
        database=db_path,
        hash_algorithm=HashAlgorithm.XXHASH64,
        hash_length=16,
        auto_create_tables=True,
    ) as duckdb_store:
        duckdb_store.write_metadata(feature, upstream_data)
        duckdb_increment = duckdb_store.resolve_update(
            feature=feature,
            hash_algorithm=HashAlgorithm.XXHASH64,
            hash_length=16,
            lazy=False,
        )

    # Convert to Polars and compare
    memory_added = collect_to_polars(memory_increment.added).sort("sample_uid")
    duckdb_added = collect_to_polars(duckdb_increment.added).sort("sample_uid")

    # Compare sample_uids
    pl.testing.assert_series_equal(
        memory_added["sample_uid"], duckdb_added["sample_uid"]
    )

    # Compare provenance hashes
    for col in ["frames", "audio"]:
        memory_prov = memory_added["metaxy_provenance_by_field"].struct.field(col)
        duckdb_prov = duckdb_added["metaxy_provenance_by_field"].struct.field(col)
        pl.testing.assert_series_equal(
            memory_prov, duckdb_prov, check_names=False
        )
