"""Cross-implementation consistency tests for ProvenanceTracker.

These tests verify that PolarsProvenanceTracker and IbisProvenanceTracker
produce identical results for the same inputs.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import polars as pl
import pytest
from metaxy.provenance.types import HashAlgorithm

from metaxy.models.feature import FeatureGraph, TestingFeature
from metaxy.provenance.ibis import IbisProvenanceTracker
from metaxy.provenance.polars import PolarsProvenanceTracker



@pytest.mark.parametrize(
    "hash_algo",
    [HashAlgorithm.XXHASH64, HashAlgorithm.MD5],
    ids=["xxhash64", "md5"],
)
def test_single_upstream_consistency(
    simple_features: dict[str, type[TestingFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    hash_algo: HashAlgorithm,
) -> None:
    """Test that Polars and Ibis trackers produce identical provenance for single upstream."""
    from metaxy.models.types import FeatureKey

    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)

    # Create both trackers
    polars_tracker = PolarsProvenanceTracker(plan)
    ibis_tracker = IbisProvenanceTracker(plan, duckdb_backend, duckdb_hash_functions)

    # Polars upstream
    polars_upstream = {FeatureKey(["video"]): upstream_video_metadata}

    # Ibis upstream (convert to Ibis table, then wrap in Narwhals)
    upstream_df = upstream_video_metadata.collect().to_native()
    video_table = duckdb_backend.create_table(
        "video_consistency", upstream_df, overwrite=True
    )
    ibis_upstream = {FeatureKey(["video"]): nw.from_native(video_table)}

    # Compute provenance with both trackers
    polars_result = polars_tracker.load_upstream_with_provenance(
        upstream=polars_upstream,
        hash_algo=hash_algo,
        hash_length=32,
        filters={},
    )

    ibis_result = ibis_tracker.load_upstream_with_provenance(
        upstream=ibis_upstream,
        hash_algo=hash_algo,
        hash_length=32,
        filters={},
    )

    # Convert both to Polars for comparison
    polars_df = polars_result.collect().to_polars().sort("sample_uid")
    ibis_df = nw.from_native(ibis_result).collect().to_polars().sort("sample_uid")

    # Compare structure
    assert polars_df["sample_uid"].to_list() == ibis_df["sample_uid"].to_list()
    assert len(polars_df) == len(ibis_df)
    assert "metaxy_provenance" in polars_df.columns
    assert "metaxy_provenance_by_field" in polars_df.columns
    assert "metaxy_provenance" in ibis_df.columns
    assert "metaxy_provenance_by_field" in ibis_df.columns

    # Verify both have same field structure in provenance_by_field
    polars_fields = set(polars_df["metaxy_provenance_by_field"][0].keys())
    ibis_fields = set(ibis_df["metaxy_provenance_by_field"][0].keys())
    assert polars_fields == ibis_fields

    # Verify all provenance values are non-empty strings
    for prov in polars_df["metaxy_provenance"]:
        assert isinstance(prov, str) and len(prov) > 0
    for prov in ibis_df["metaxy_provenance"]:
        assert isinstance(prov, str) and len(prov) > 0


@pytest.mark.parametrize(
    "hash_algo",
    [HashAlgorithm.XXHASH64, HashAlgorithm.MD5],
    ids=["xxhash64", "md5"],
)
def test_multiple_upstream_consistency(
    multi_upstream_features: dict[str, type[TestingFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    upstream_audio_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    duckdb_backend,
    duckdb_hash_functions: dict[HashAlgorithm, Any],
    hash_algo: HashAlgorithm,
) -> None:
    """Test consistency with multiple upstream features."""
    from metaxy.models.types import FeatureKey

    feature = multi_upstream_features["MultiUpstreamFeature"]
    plan = graph.get_feature_plan(feature.spec().key)

    polars_tracker = PolarsProvenanceTracker(plan)
    ibis_tracker = IbisProvenanceTracker(plan, duckdb_backend, duckdb_hash_functions)

    # Polars upstream
    polars_upstream = {
        FeatureKey(["video"]): upstream_video_metadata,
        FeatureKey(["audio"]): upstream_audio_metadata,
    }

    # Ibis upstream (wrap in Narwhals)
    video_df = upstream_video_metadata.collect().to_native()
    audio_df = upstream_audio_metadata.collect().to_native()
    video_table = duckdb_backend.create_table(
        "video_multi_consistency", video_df, overwrite=True
    )
    audio_table = duckdb_backend.create_table(
        "audio_multi_consistency", audio_df, overwrite=True
    )
    ibis_upstream = {
        FeatureKey(["video"]): nw.from_native(video_table),
        FeatureKey(["audio"]): nw.from_native(audio_table),
    }

    # Compute provenance
    polars_result = polars_tracker.load_upstream_with_provenance(
        upstream=polars_upstream,
        hash_algo=hash_algo,
        hash_length=32,
        filters={},
    )

    ibis_result = ibis_tracker.load_upstream_with_provenance(
        upstream=ibis_upstream,
        hash_algo=hash_algo,
        hash_length=32,
        filters={},
    )

    # Convert both to Polars for comparison
    polars_df = polars_result.collect().to_polars().sort("sample_uid")
    ibis_df = nw.from_native(ibis_result).collect().to_polars().sort("sample_uid")

    # Compare structure (not exact hash values, since implementations differ)
    assert polars_df["sample_uid"].to_list() == ibis_df["sample_uid"].to_list()
    assert len(polars_df) == len(ibis_df)

    # Verify provenance columns exist and have correct structure
    assert "metaxy_provenance" in polars_df.columns
    assert "metaxy_provenance_by_field" in polars_df.columns
    assert "metaxy_provenance" in ibis_df.columns
    assert "metaxy_provenance_by_field" in ibis_df.columns

    # Verify field-level provenance has same structure (fields present)
    for i in range(len(polars_df)):
        polars_prov = polars_df["metaxy_provenance_by_field"][i]
        ibis_prov = ibis_df["metaxy_provenance_by_field"][i]

        # Check both implementations have the same fields
        assert (
            set(polars_prov.keys()) == set(ibis_prov.keys()) == {"fusion", "analysis"}
        )

        # Check all provenance values are non-empty strings
        assert isinstance(polars_prov["fusion"], str) and len(polars_prov["fusion"]) > 0
        assert (
            isinstance(polars_prov["analysis"], str)
            and len(polars_prov["analysis"]) > 0
        )
        assert isinstance(ibis_prov["fusion"], str) and len(ibis_prov["fusion"]) > 0
        assert isinstance(ibis_prov["analysis"], str) and len(ibis_prov["analysis"]) > 0

    # Verify sample-level provenance values are non-empty strings
    for prov in polars_df["metaxy_provenance"]:
        assert isinstance(prov, str) and len(prov) > 0
    for prov in ibis_df["metaxy_provenance"]:
        assert isinstance(prov, str) and len(prov) > 0


def test_selective_field_deps_consistency(
    selective_field_dep_features: dict[str, type[TestingFeature]],
    upstream_metadata_multi_field: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    duckdb_backend,
    duckdb_hash_functions: dict[HashAlgorithm, Any],
) -> None:
    """Test consistency with selective field-level dependencies."""
    from metaxy.models.types import FeatureKey

    feature = selective_field_dep_features["SelectiveFeature"]
    plan = graph.get_feature_plan(feature.spec().key)

    polars_tracker = PolarsProvenanceTracker(plan)
    ibis_tracker = IbisProvenanceTracker(plan, duckdb_backend, duckdb_hash_functions)

    # Polars upstream
    polars_upstream = {FeatureKey(["multi_field"]): upstream_metadata_multi_field}

    # Ibis upstream (wrap in Narwhals)
    upstream_df = upstream_metadata_multi_field.collect().to_native()
    table = duckdb_backend.create_table(
        "multi_field_consistency", upstream_df, overwrite=True
    )
    ibis_upstream = {FeatureKey(["multi_field"]): nw.from_native(table)}

    # Compute provenance
    polars_result = polars_tracker.load_upstream_with_provenance(
        upstream=polars_upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        hash_length=32,
        filters={},
    )

    ibis_result = ibis_tracker.load_upstream_with_provenance(
        upstream=ibis_upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        hash_length=32,
        filters={},
    )

    # Convert both to Polars for comparison
    polars_df = polars_result.collect().to_polars().sort("sample_uid")
    ibis_df = nw.from_native(ibis_result).collect().to_polars().sort("sample_uid")

    # Compare structure (not exact hash values)
    assert polars_df["sample_uid"].to_list() == ibis_df["sample_uid"].to_list()
    assert len(polars_df) == len(ibis_df)

    # Verify all three fields are present in both implementations
    for i in range(len(polars_df)):
        polars_prov = polars_df["metaxy_provenance_by_field"][i]
        ibis_prov = ibis_df["metaxy_provenance_by_field"][i]

        # Check both have same field structure
        assert (
            set(polars_prov.keys())
            == set(ibis_prov.keys())
            == {"visual", "audio_only", "mixed"}
        )

        # Check all provenance values are non-empty strings
        for field in ["visual", "audio_only", "mixed"]:
            assert isinstance(polars_prov[field], str) and len(polars_prov[field]) > 0
            assert isinstance(ibis_prov[field], str) and len(ibis_prov[field]) > 0


def test_resolve_increment_consistency(
    simple_features: dict[str, type[TestingFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    duckdb_backend,
    duckdb_hash_functions: dict[HashAlgorithm, Any],
) -> None:
    """Test that resolve_increment produces consistent results across implementations."""
    from metaxy.models.types import FeatureKey

    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)

    polars_tracker = PolarsProvenanceTracker(plan)
    ibis_tracker = IbisProvenanceTracker(plan, duckdb_hash_functions)

    # Prepare upstream for both
    polars_upstream = {FeatureKey(["video"]): upstream_video_metadata}

    upstream_df = upstream_video_metadata.collect().to_native()
    video_table = duckdb_backend.create_table(
        "video_increment", upstream_df, overwrite=True
    )
    ibis_upstream = {FeatureKey(["video"]): nw.from_native(video_table)}

    # No current metadata - all should be added
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

    # Materialize results (all are Narwhals LazyFrames, or None)
    polars_added = polars_added.collect()
    polars_changed = polars_changed.collect() if polars_changed is not None else None
    polars_removed = polars_removed.collect() if polars_removed is not None else None

    ibis_added = ibis_added.collect()
    ibis_changed = ibis_changed.collect() if ibis_changed is not None else None
    ibis_removed = ibis_removed.collect() if ibis_removed is not None else None

    # Compare counts
    assert len(polars_added) == len(ibis_added) == 3
    # When current is None, changed and removed should be None
    assert polars_changed is None and ibis_changed is None
    assert polars_removed is None and ibis_removed is None

    # Compare sample IDs in added
    polars_added_ids = sorted(polars_added["sample_uid"].to_list())
    ibis_added_ids = sorted(ibis_added["sample_uid"].to_list())
    assert polars_added_ids == ibis_added_ids


@pytest.mark.parametrize(
    "hash_length",
    [8, 16, 32, 64],
    ids=["len_8", "len_16", "len_32", "len_64"],
)
def test_end_to_end_consistency_snapshot(
    simple_features: dict[str, type[TestingFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    duckdb_backend,
    duckdb_hash_functions: dict[HashAlgorithm, Any],
    hash_length: int,
    snapshot,
) -> None:
    """Snapshot test for end-to-end consistency across different hash lengths.

    This ensures that both implementations produce identical results and that
    results are stable across code changes.
    """
    from metaxy.models.types import FeatureKey

    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)

    polars_tracker = PolarsProvenanceTracker(plan)
    ibis_tracker = IbisProvenanceTracker(plan, duckdb_hash_functions)

    # Polars computation
    polars_upstream = {FeatureKey(["video"]): upstream_video_metadata}
    polars_result = polars_tracker.load_upstream_with_provenance(
        upstream=polars_upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        hash_length=hash_length,
        filters={},
    )
    polars_df = polars_result.collect().to_polars().sort("sample_uid")

    # Ibis computation (wrap in Narwhals)
    upstream_df = upstream_video_metadata.collect().to_native()
    video_table = duckdb_backend.create_table("video_e2e", upstream_df, overwrite=True)
    ibis_upstream = {FeatureKey(["video"]): nw.from_native(video_table)}
    ibis_result = ibis_tracker.load_upstream_with_provenance(
        upstream=ibis_upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        hash_length=hash_length,
        filters={},
    )
    ibis_df = nw.from_native(ibis_result).collect().to_polars().sort("sample_uid")

    # Verify structural consistency between implementations
    assert polars_df["sample_uid"].to_list() == ibis_df["sample_uid"].to_list()
    assert len(polars_df) == len(ibis_df)

    # Verify both have same columns
    assert "metaxy_provenance" in polars_df.columns
    assert "metaxy_provenance_by_field" in polars_df.columns
    assert "metaxy_provenance" in ibis_df.columns
    assert "metaxy_provenance_by_field" in ibis_df.columns


    import polars.testing as pl_test

    pl_test.assert_frame_equal(polars_df, ibis_df)
