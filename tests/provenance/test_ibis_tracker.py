"""Unit tests for IbisProvenanceTracker."""

from __future__ import annotations

from typing import Any

import narwhals as nw
import polars as pl
import pytest

from metaxy.data_versioning import HashAlgorithm
from metaxy.models.feature import FeatureGraph, TestingFeature
from metaxy.provenance.ibis import IbisProvenanceTracker


@pytest.fixture
def duckdb_backend():
    """Create a DuckDB backend for testing."""
    import ibis

    return ibis.duckdb.connect()


@pytest.fixture
def duckdb_hash_functions() -> dict[HashAlgorithm, Any]:
    """Create hash functions for DuckDB backend.

    Note: For testing, these functions just cast to string without actual hashing.
    This is sufficient to test the provenance tracking logic.
    Production code uses DuckDBProvenanceByFieldCalculator which generates proper SQL.
    """

    def xxhash64(col_expr):
        # For testing, just cast to string (no actual hashing)
        # This tests provenance logic without needing DuckDB hash extensions
        return col_expr.cast(str)

    def md5_hash(col_expr):
        # For testing, just cast to string
        return col_expr.cast(str)

    def sha256_hash(col_expr):
        # For testing, just cast to string
        return col_expr.cast(str)

    return {
        HashAlgorithm.XXHASH64: xxhash64,
        HashAlgorithm.MD5: md5_hash,
        HashAlgorithm.SHA256: sha256_hash,
    }


def test_ibis_tracker_initialization(
    simple_features: dict[str, type[TestingFeature]],
    graph: FeatureGraph,
    duckdb_backend,
    duckdb_hash_functions: dict[HashAlgorithm, Any],
) -> None:
    """Test IbisProvenanceTracker can be initialized."""
    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = IbisProvenanceTracker(plan, duckdb_backend, duckdb_hash_functions)
    assert tracker.plan is plan
    assert tracker.backend is duckdb_backend
    assert tracker.hash_functions is duckdb_hash_functions


def test_compute_provenance_single_upstream(
    simple_features: dict[str, type[TestingFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    duckdb_backend,
    duckdb_hash_functions: dict[HashAlgorithm, Any],
    snapshot,
) -> None:
    """Test load_upstream_with_provenance with single upstream feature."""
    from metaxy.models.types import FeatureKey

    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = IbisProvenanceTracker(plan, duckdb_backend, duckdb_hash_functions)

    # Convert Polars LazyFrame to Ibis table, then wrap in Narwhals
    upstream_df = upstream_video_metadata.collect().to_native()
    video_table = duckdb_backend.create_table(
        "video_upstream", upstream_df, overwrite=True
    )

    upstream = {FeatureKey(["video"]): nw.from_native(video_table)}

    result = tracker.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        hash_length=32,
        filters={},
    )

    # Materialize for inspection
    result_df = nw.from_native(result).collect()

    # Verify provenance columns exist
    assert "metaxy_provenance_by_field" in result_df.columns
    assert "metaxy_provenance" in result_df.columns

    # Verify provenance_by_field has the expected field
    first_prov = result_df["metaxy_provenance_by_field"][0]
    assert "default" in first_prov

    # Snapshot the provenance values
    provenance_data = []
    for i in range(len(result_df)):
        provenance_data.append(
            {
                "sample_uid": result_df["sample_uid"][i],
                "field_provenance": result_df["metaxy_provenance_by_field"][i],
                "sample_provenance": result_df["metaxy_provenance"][i],
            }
        )

    assert provenance_data == snapshot


def test_compute_provenance_multiple_upstreams(
    multi_upstream_features: dict[str, type[TestingFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    upstream_audio_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    duckdb_backend,
    duckdb_hash_functions: dict[HashAlgorithm, Any],
    snapshot,
) -> None:
    """Test load_upstream_with_provenance with multiple upstream features."""
    from metaxy.models.types import FeatureKey

    feature = multi_upstream_features["MultiUpstreamFeature"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = IbisProvenanceTracker(plan, duckdb_backend, duckdb_hash_functions)

    # Convert to Ibis tables, then wrap in Narwhals
    video_df = upstream_video_metadata.collect().to_native()
    audio_df = upstream_audio_metadata.collect().to_native()
    video_table = duckdb_backend.create_table("video_multi", video_df, overwrite=True)
    audio_table = duckdb_backend.create_table("audio_multi", audio_df, overwrite=True)

    upstream = {
        FeatureKey(["video"]): nw.from_native(video_table),
        FeatureKey(["audio"]): nw.from_native(audio_table),
    }

    result = tracker.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        hash_length=32,
        filters={},
    )

    result_df = nw.from_native(result).collect()

    # Verify provenance columns exist
    assert "metaxy_provenance_by_field" in result_df.columns
    assert "metaxy_provenance" in result_df.columns

    # Verify provenance_by_field has both fields
    first_prov = result_df["metaxy_provenance_by_field"][0]
    assert "fusion" in first_prov
    assert "analysis" in first_prov

    # Different code versions should produce different field hashes
    assert first_prov["fusion"] != first_prov["analysis"]

    # Snapshot the results
    provenance_data = []
    for i in range(len(result_df)):
        provenance_data.append(
            {
                "sample_uid": result_df["sample_uid"][i],
                "field_provenance": result_df["metaxy_provenance_by_field"][i],
                "sample_provenance": result_df["metaxy_provenance"][i],
            }
        )

    assert provenance_data == snapshot


def test_resolve_increment_no_current(
    simple_features: dict[str, type[TestingFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    duckdb_backend,
    duckdb_hash_functions: dict[HashAlgorithm, Any],
) -> None:
    """Test resolve_increment_with_provenance when no current metadata exists."""
    from metaxy.models.types import FeatureKey

    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = IbisProvenanceTracker(plan, duckdb_backend, duckdb_hash_functions)

    # Convert to Ibis table, then wrap in Narwhals
    upstream_df = upstream_video_metadata.collect().to_native()
    video_table = duckdb_backend.create_table(
        "video_no_current", upstream_df, overwrite=True
    )

    upstream = {FeatureKey(["video"]): nw.from_native(video_table)}

    added, changed, removed = tracker.resolve_increment_with_provenance(
        current=None,
        upstream=upstream,
        hash_algorithm=HashAlgorithm.XXHASH64,
        hash_length=32,
        filters={},
        sample=None,
    )

    # Materialize results (they are already Narwhals LazyFrames, or None)
    added_df = added.collect()
    changed_df = changed.collect() if changed is not None else None
    removed_df = removed.collect() if removed is not None else None

    # All samples should be added
    assert len(added_df) == 3
    assert changed_df is None
    assert removed_df is None

    # Verify added has the correct columns
    assert "sample_uid" in added_df.columns
    assert "metaxy_provenance" in added_df.columns


def test_resolve_increment_with_changes(
    simple_features: dict[str, type[TestingFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    duckdb_backend,
    duckdb_hash_functions: dict[HashAlgorithm, Any],
) -> None:
    """Test resolve_increment_with_provenance identifies added, changed, removed."""
    from metaxy.models.types import FeatureKey

    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = IbisProvenanceTracker(plan, duckdb_backend, duckdb_hash_functions)

    # Convert to Ibis table, then wrap in Narwhals
    upstream_df = upstream_video_metadata.collect().to_native()
    video_table = duckdb_backend.create_table(
        "video_changes", upstream_df, overwrite=True
    )

    upstream = {FeatureKey(["video"]): nw.from_native(video_table)}

    # First, compute expected provenance
    expected = tracker.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        hash_length=32,
        filters={},
    )
    expected_df = nw.from_native(expected).collect()

    # Create current metadata with changes
    current_pl = pl.DataFrame(
        {
            "sample_uid": [1, 2, 4],
            "metaxy_provenance_by_field": [
                expected_df["metaxy_provenance_by_field"][0],  # Same as expected
                {"default": "different_hash"},  # Different
                {"default": "removed_hash"},  # Will be removed
            ],
            "metaxy_provenance": [
                expected_df["metaxy_provenance"][0],  # Same
                "different_provenance",  # Different
                "removed_provenance",  # Removed
            ],
        }
    )
    current_table = duckdb_backend.create_table(
        "current_changes", current_pl, overwrite=True
    )

    added, changed, removed = tracker.resolve_increment_with_provenance(
        current=nw.from_native(current_table),
        upstream=upstream,
        hash_algorithm=HashAlgorithm.XXHASH64,
        hash_length=32,
        filters={},
        sample=None,
    )

    # Materialize results (they are already Narwhals LazyFrames)
    added_df = added.collect()
    assert changed is not None
    assert removed is not None
    changed_df = changed.collect()
    removed_df = removed.collect()

    # Verify counts
    assert len(added_df) == 1  # sample 3
    assert len(changed_df) == 1  # sample 2
    assert len(removed_df) == 1  # sample 4

    # Verify sample IDs
    assert added_df["sample_uid"][0] == 3
    assert changed_df["sample_uid"][0] == 2
    assert removed_df["sample_uid"][0] == 4


def test_stays_lazy_until_collection(
    simple_features: dict[str, type[TestingFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    duckdb_backend,
    duckdb_hash_functions: dict[HashAlgorithm, Any],
) -> None:
    """Test that Ibis tracker stays lazy and only executes on explicit collection."""
    import ibis

    from metaxy.models.types import FeatureKey

    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = IbisProvenanceTracker(plan, duckdb_backend, duckdb_hash_functions)

    # Convert to Ibis table, then wrap in Narwhals
    upstream_df = upstream_video_metadata.collect().to_native()
    video_table = duckdb_backend.create_table("video_lazy", upstream_df, overwrite=True)

    upstream = {FeatureKey(["video"]): nw.from_native(video_table)}

    # Compute provenance - should return Narwhals LazyFrame wrapping Ibis table
    result = tracker.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        hash_length=32,
        filters={},
    )

    # Result should be Narwhals LazyFrame wrapping an Ibis table (lazy)
    assert isinstance(result, nw.LazyFrame)
    # Get the underlying Ibis table
    native_result = result.to_native()
    assert isinstance(native_result, ibis.expr.types.relations.Table)

    # Only when we collect should it materialize
    result_df = result.collect()
    assert len(result_df) == 3


def test_prepare_upstream_applies_filters(
    simple_features: dict[str, type[TestingFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    duckdb_backend,
    duckdb_hash_functions: dict[HashAlgorithm, Any],
) -> None:
    """Test prepare_upstream correctly applies filters."""
    from metaxy.models.types import FeatureKey

    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = IbisProvenanceTracker(plan, duckdb_backend, duckdb_hash_functions)

    # Convert to Ibis table, then wrap in Narwhals
    upstream_df = upstream_video_metadata.collect().to_native()
    video_table = duckdb_backend.create_table(
        "video_filtered", upstream_df, overwrite=True
    )

    upstream = {FeatureKey(["video"]): nw.from_native(video_table)}

    # Filter to only include sample_uid > 1
    filters = {FeatureKey(["video"]): [nw.col("sample_uid") > 1]}

    prepared = tracker.prepare_upstream(
        upstream=upstream,
        filters=filters,
    )

    prepared_df = nw.from_native(prepared).collect()

    # Only samples 2 and 3 should remain
    assert len(prepared_df) == 2
    assert set(prepared_df["sample_uid"].to_list()) == {2, 3}
