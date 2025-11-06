"""Comparison tests between PolarsProvenanceTracker and IbisProvenanceTracker (DuckDB).

These tests verify that both tracker implementations produce identical results
for resolve_increment_with_provenance, which is the core method used by the system.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import polars as pl
import pytest

from metaxy._utils import collect_to_polars
from metaxy.models.feature import FeatureGraph, TestingFeature
from metaxy.models.types import FeatureKey
from metaxy.provenance.ibis import IbisProvenanceTracker
from metaxy.provenance.polars import PolarsProvenanceTracker
from metaxy.provenance.types import HashAlgorithm


@pytest.fixture
def duckdb_connection():
    """Create a DuckDB connection for Ibis tracker tests."""
    import ibis

    conn = ibis.duckdb.connect()
    # Load hashfuncs extension
    conn.raw_sql("INSTALL hashfuncs FROM community")
    conn.raw_sql("LOAD hashfuncs")
    return conn


@pytest.fixture
def duckdb_hash_functions() -> dict[HashAlgorithm, Any]:
    """Create hash functions compatible with both Polars and DuckDB.

    These are designed to produce consistent results across implementations.
    """
    import ibis

    @ibis.udf.scalar.builtin
    def MD5(x: str) -> str: ...

    @ibis.udf.scalar.builtin
    def HEX(x: str) -> str: ...

    @ibis.udf.scalar.builtin
    def LOWER(x: str) -> str: ...

    @ibis.udf.scalar.builtin
    def xxh32(x: str) -> str: ...

    @ibis.udf.scalar.builtin
    def xxh64(x: str) -> str: ...

    def md5_hash(expr):
        return LOWER(HEX(MD5(expr.cast(str))))

    def xxhash32_hash(expr):
        return xxh32(expr).cast(str)

    def xxhash64_hash(expr):
        return xxh64(expr).cast(str)

    return {
        HashAlgorithm.MD5: md5_hash,
        HashAlgorithm.XXHASH32: xxhash32_hash,
        HashAlgorithm.XXHASH64: xxhash64_hash,
    }


def test_resolve_increment_no_current_metadata(
    simple_features: dict[str, type[TestingFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    duckdb_connection,
    duckdb_hash_functions: dict[HashAlgorithm, Any],
) -> None:
    """Test resolve_increment when there's no current metadata (all samples are added).

    This is the simplest case - both trackers should return identical added samples.
    """
    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)

    # Create both trackers
    polars_tracker = PolarsProvenanceTracker(plan)
    ibis_tracker = IbisProvenanceTracker(plan, duckdb_hash_functions)

    # Polars upstream
    polars_upstream = {FeatureKey(["video"]): upstream_video_metadata}

    # Ibis upstream - convert to Ibis table
    upstream_df = upstream_video_metadata.collect().to_native()
    video_table = duckdb_connection.create_table(
        "video_no_current", upstream_df, overwrite=True
    )
    ibis_upstream = {FeatureKey(["video"]): nw.from_native(video_table)}

    # Resolve increment with no current metadata
    polars_added, polars_changed, polars_removed = (
        polars_tracker.resolve_increment_with_provenance(
            current=None,
            upstream=polars_upstream,
            hash_algorithm=HashAlgorithm.XXHASH64,
            hash_length=32,
            filters={},
            sample=None,
        )
    )

    ibis_added, ibis_changed, ibis_removed = (
        ibis_tracker.resolve_increment_with_provenance(
            current=None,
            upstream=ibis_upstream,
            hash_algorithm=HashAlgorithm.XXHASH64,
            hash_length=32,
            filters={},
            sample=None,
        )
    )

    # Materialize results
    polars_added_df = collect_to_polars(polars_added).sort("sample_uid")
    ibis_added_df = collect_to_polars(ibis_added).sort("sample_uid")

    # Both should have same samples added
    assert len(polars_added_df) == len(ibis_added_df) == 3
    assert polars_added_df["sample_uid"].to_list() == ibis_added_df["sample_uid"].to_list()

    # Changed and removed should be None
    assert polars_changed is None and ibis_changed is None
    assert polars_removed is None and ibis_removed is None

    # Both should have provenance columns
    assert "metaxy_provenance" in polars_added_df.columns
    assert "metaxy_provenance_by_field" in polars_added_df.columns
    assert "metaxy_provenance" in ibis_added_df.columns
    assert "metaxy_provenance_by_field" in ibis_added_df.columns

    # Verify field structure is identical
    for i in range(len(polars_added_df)):
        polars_fields = set(polars_added_df["metaxy_provenance_by_field"][i].keys())
        ibis_fields = set(ibis_added_df["metaxy_provenance_by_field"][i].keys())
        assert polars_fields == ibis_fields, f"Field mismatch at row {i}"


def test_resolve_increment_with_changes(
    simple_features: dict[str, type[TestingFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    duckdb_connection,
    duckdb_hash_functions: dict[HashAlgorithm, Any],
) -> None:
    """Test resolve_increment when some samples are added, changed, and removed."""
    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)

    polars_tracker = PolarsProvenanceTracker(plan)
    ibis_tracker = IbisProvenanceTracker(plan, duckdb_hash_functions)

    # Create current metadata with samples 1 and 2 (sample 3 will be added)
    current_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {"default": "old_hash_1"},  # Will be changed
                    {"default": "matching_hash"},  # Needs to be computed to match
                ],
                "metaxy_provenance": ["old_prov_1", "old_prov_2"],
            }
        ).lazy()
    )

    # Polars computation
    polars_upstream = {FeatureKey(["video"]): upstream_video_metadata}
    polars_added, polars_changed, polars_removed = (
        polars_tracker.resolve_increment_with_provenance(
            current=current_metadata,
            upstream=polars_upstream,
            hash_algorithm=HashAlgorithm.XXHASH64,
            hash_length=32,
            filters={},
            sample=None,
        )
    )

    # Ibis computation
    upstream_df = upstream_video_metadata.collect().to_native()
    video_table = duckdb_connection.create_table(
        "video_with_changes", upstream_df, overwrite=True
    )
    current_df = current_metadata.collect().to_native()
    current_table = duckdb_connection.create_table(
        "current_with_changes", current_df, overwrite=True
    )
    ibis_upstream = {FeatureKey(["video"]): nw.from_native(video_table)}
    ibis_current = nw.from_native(current_table)

    ibis_added, ibis_changed, ibis_removed = (
        ibis_tracker.resolve_increment_with_provenance(
            current=ibis_current,
            upstream=ibis_upstream,
            hash_algorithm=HashAlgorithm.XXHASH64,
            hash_length=32,
            filters={},
            sample=None,
        )
    )

    # Materialize results
    polars_added_df = collect_to_polars(polars_added).sort("sample_uid")
    ibis_added_df = collect_to_polars(ibis_added).sort("sample_uid")

    # Sample 3 should be added in both
    assert len(polars_added_df) == len(ibis_added_df) == 1
    assert polars_added_df["sample_uid"].to_list() == ibis_added_df["sample_uid"].to_list() == [3]

    # Both should have changed samples (samples 1 and 2 with mismatched provenance)
    assert polars_changed is not None and ibis_changed is not None
    polars_changed_df = collect_to_polars(polars_changed).sort("sample_uid")
    ibis_changed_df = collect_to_polars(ibis_changed).sort("sample_uid")

    # At least sample 1 should be changed (we gave it an old hash)
    assert len(polars_changed_df) >= 1
    assert len(ibis_changed_df) >= 1
    assert 1 in polars_changed_df["sample_uid"].to_list()
    assert 1 in ibis_changed_df["sample_uid"].to_list()

    # Compare the changed samples
    assert set(polars_changed_df["sample_uid"].to_list()) == set(
        ibis_changed_df["sample_uid"].to_list()
    )

    # Both should have no removed samples (all samples in current are in upstream)
    # Note: removed may be an empty DataFrame or None
    if polars_removed is not None:
        polars_removed_df = collect_to_polars(polars_removed)
        assert len(polars_removed_df) == 0
    if ibis_removed is not None:
        ibis_removed_df = collect_to_polars(ibis_removed)
        assert len(ibis_removed_df) == 0


def test_resolve_increment_with_removal(
    simple_features: dict[str, type[TestingFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    duckdb_connection,
    duckdb_hash_functions: dict[HashAlgorithm, Any],
) -> None:
    """Test resolve_increment when samples are removed from upstream."""
    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)

    polars_tracker = PolarsProvenanceTracker(plan)
    ibis_tracker = IbisProvenanceTracker(plan, duckdb_hash_functions)

    # Create current metadata with samples 1, 2, 3, and 4
    # Upstream only has 1, 2, 3 - so sample 4 will be removed
    current_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3, 4],
                "metaxy_provenance_by_field": [
                    {"default": "hash_1"},
                    {"default": "hash_2"},
                    {"default": "hash_3"},
                    {"default": "hash_4"},
                ],
                "metaxy_provenance": ["prov_1", "prov_2", "prov_3", "prov_4"],
            }
        ).lazy()
    )

    # Polars computation
    polars_upstream = {FeatureKey(["video"]): upstream_video_metadata}
    polars_added, polars_changed, polars_removed = (
        polars_tracker.resolve_increment_with_provenance(
            current=current_metadata,
            upstream=polars_upstream,
            hash_algorithm=HashAlgorithm.XXHASH64,
            hash_length=32,
            filters={},
            sample=None,
        )
    )

    # Ibis computation
    upstream_df = upstream_video_metadata.collect().to_native()
    video_table = duckdb_connection.create_table(
        "video_with_removal", upstream_df, overwrite=True
    )
    current_df = current_metadata.collect().to_native()
    current_table = duckdb_connection.create_table(
        "current_with_removal", current_df, overwrite=True
    )
    ibis_upstream = {FeatureKey(["video"]): nw.from_native(video_table)}
    ibis_current = nw.from_native(current_table)

    ibis_added, ibis_changed, ibis_removed = (
        ibis_tracker.resolve_increment_with_provenance(
            current=ibis_current,
            upstream=ibis_upstream,
            hash_algorithm=HashAlgorithm.XXHASH64,
            hash_length=32,
            filters={},
            sample=None,
        )
    )

    # Materialize results
    polars_added_df = collect_to_polars(polars_added)
    ibis_added_df = collect_to_polars(ibis_added)

    # No new samples should be added
    assert len(polars_added_df) == len(ibis_added_df) == 0

    # Sample 4 should be removed in both
    assert polars_removed is not None and ibis_removed is not None
    polars_removed_df = collect_to_polars(polars_removed).sort("sample_uid")
    ibis_removed_df = collect_to_polars(ibis_removed).sort("sample_uid")

    assert len(polars_removed_df) == len(ibis_removed_df) == 1
    assert polars_removed_df["sample_uid"].to_list() == ibis_removed_df["sample_uid"].to_list() == [4]

    # Verify removed samples have provenance columns
    assert "metaxy_provenance" in polars_removed_df.columns
    assert "metaxy_provenance_by_field" in polars_removed_df.columns
    assert "metaxy_provenance" in ibis_removed_df.columns
    assert "metaxy_provenance_by_field" in ibis_removed_df.columns


def test_resolve_increment_multi_upstream(
    multi_upstream_features: dict[str, type[TestingFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    upstream_audio_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    duckdb_connection,
    duckdb_hash_functions: dict[HashAlgorithm, Any],
) -> None:
    """Test resolve_increment with multiple upstream features."""
    feature = multi_upstream_features["MultiUpstreamFeature"]
    plan = graph.get_feature_plan(feature.spec().key)

    polars_tracker = PolarsProvenanceTracker(plan)
    ibis_tracker = IbisProvenanceTracker(plan, duckdb_hash_functions)

    # Polars upstream
    polars_upstream = {
        FeatureKey(["video"]): upstream_video_metadata,
        FeatureKey(["audio"]): upstream_audio_metadata,
    }

    # Ibis upstream
    video_df = upstream_video_metadata.collect().to_native()
    audio_df = upstream_audio_metadata.collect().to_native()
    video_table = duckdb_connection.create_table(
        "video_multi", video_df, overwrite=True
    )
    audio_table = duckdb_connection.create_table(
        "audio_multi", audio_df, overwrite=True
    )
    ibis_upstream = {
        FeatureKey(["video"]): nw.from_native(video_table),
        FeatureKey(["audio"]): nw.from_native(audio_table),
    }

    # Resolve increment with no current metadata
    polars_added, polars_changed, polars_removed = (
        polars_tracker.resolve_increment_with_provenance(
            current=None,
            upstream=polars_upstream,
            hash_algorithm=HashAlgorithm.XXHASH64,
            hash_length=32,
            filters={},
            sample=None,
        )
    )

    ibis_added, ibis_changed, ibis_removed = (
        ibis_tracker.resolve_increment_with_provenance(
            current=None,
            upstream=ibis_upstream,
            hash_algorithm=HashAlgorithm.XXHASH64,
            hash_length=32,
            filters={},
            sample=None,
        )
    )

    # Materialize results
    polars_added_df = collect_to_polars(polars_added).sort("sample_uid")
    ibis_added_df = collect_to_polars(ibis_added).sort("sample_uid")

    # Both should have same samples
    assert len(polars_added_df) == len(ibis_added_df) == 3
    assert polars_added_df["sample_uid"].to_list() == ibis_added_df["sample_uid"].to_list()

    # Changed and removed should be None
    assert polars_changed is None and ibis_changed is None
    assert polars_removed is None and ibis_removed is None

    # Verify field structure is identical (should have "fusion" and "analysis" fields)
    for i in range(len(polars_added_df)):
        polars_fields = set(polars_added_df["metaxy_provenance_by_field"][i].keys())
        ibis_fields = set(ibis_added_df["metaxy_provenance_by_field"][i].keys())
        assert polars_fields == ibis_fields == {"fusion", "analysis"}


@pytest.mark.parametrize(
    "hash_algo",
    [HashAlgorithm.XXHASH64, HashAlgorithm.XXHASH32, HashAlgorithm.MD5],
    ids=["xxhash64", "xxhash32", "md5"],
)
def test_resolve_increment_hash_consistency(
    simple_features: dict[str, type[TestingFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    duckdb_connection,
    duckdb_hash_functions: dict[HashAlgorithm, Any],
    hash_algo: HashAlgorithm,
) -> None:
    """Test that resolve_increment produces consistent results across hash algorithms."""
    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)

    polars_tracker = PolarsProvenanceTracker(plan)
    ibis_tracker = IbisProvenanceTracker(plan, duckdb_hash_functions)

    # Polars upstream
    polars_upstream = {FeatureKey(["video"]): upstream_video_metadata}

    # Ibis upstream
    upstream_df = upstream_video_metadata.collect().to_native()
    video_table = duckdb_connection.create_table(
        f"video_hash_{hash_algo.value}", upstream_df, overwrite=True
    )
    ibis_upstream = {FeatureKey(["video"]): nw.from_native(video_table)}

    # Resolve increment
    polars_added, _, _ = polars_tracker.resolve_increment_with_provenance(
        current=None,
        upstream=polars_upstream,
        hash_algorithm=hash_algo,
        hash_length=32,
        filters={},
        sample=None,
    )

    ibis_added, _, _ = ibis_tracker.resolve_increment_with_provenance(
        current=None,
        upstream=ibis_upstream,
        hash_algorithm=hash_algo,
        hash_length=32,
        filters={},
        sample=None,
    )

    # Materialize
    polars_added_df = collect_to_polars(polars_added).sort("sample_uid")
    ibis_added_df = collect_to_polars(ibis_added).sort("sample_uid")

    # Both should have same samples
    assert len(polars_added_df) == len(ibis_added_df) == 3
    assert polars_added_df["sample_uid"].to_list() == ibis_added_df["sample_uid"].to_list()

    # Both should have provenance with consistent structure
    for i in range(len(polars_added_df)):
        polars_prov = polars_added_df["metaxy_provenance_by_field"][i]
        ibis_prov = ibis_added_df["metaxy_provenance_by_field"][i]

        # Same fields should be present
        assert set(polars_prov.keys()) == set(ibis_prov.keys())

        # All provenance values should be non-empty strings
        for field in polars_prov.keys():
            assert isinstance(polars_prov[field], str) and len(polars_prov[field]) > 0
            assert isinstance(ibis_prov[field], str) and len(ibis_prov[field]) > 0
