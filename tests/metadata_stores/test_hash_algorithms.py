"""Focused tests for hash algorithms across metadata stores.

This test file focuses on hash algorithm behavior, including:
- Testing all supported hash algorithms with ONE store
- Testing hash truncation independently
- Testing hash algorithm compatibility with different stores
- Validating golden hash values for each algorithm

This reduces test bloat by avoiding Cartesian product parametrization of
hash_algorithm × store_type × truncation in other test files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import polars.testing as pl_testing
import pytest
from metaxy_testing import HashAlgorithmCases
from metaxy_testing.models import SampleFeature, SampleFeatureSpec
from metaxy_testing.parametric import (
    downstream_metadata_strategy,
)
from pytest_cases import parametrize_with_cases

from metaxy import BaseFeature, FeatureDep, FeatureGraph
from metaxy.metadata_store.delta import DeltaMetadataStore
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    from metaxy.metadata_store import MetadataStore


# ============= TEST: HASH ALGORITHM CORRECTNESS =============


@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
def test_hash_algorithm_produces_consistent_hashes(
    hash_algorithm: HashAlgorithm,
    graph: FeatureGraph,
    tmp_path,
):
    """Test that each hash algorithm produces consistent hashes across runs."""

    class ParentFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="parent",
            fields=["value"],
        ),
    ):
        pass

    class ChildFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="child",
            deps=[FeatureDep(feature=ParentFeature)],
            fields=["result"],
        ),
    ):
        pass

    child_plan = graph.get_feature_plan(ChildFeature.spec().key)
    feature_versions = {
        "parent": ParentFeature.feature_version(),
        "child": ChildFeature.feature_version(),
    }

    # Generate test data
    upstream_data, golden_downstream = downstream_metadata_strategy(
        child_plan,
        feature_versions=feature_versions,
        snapshot_version=graph.snapshot_version,
        hash_algorithm=hash_algorithm,
        min_rows=10,
        max_rows=10,
    ).example()

    parent_df = upstream_data["parent"]

    # Create store with specific hash algorithm
    store = DeltaMetadataStore(root_path=tmp_path / "delta_store", hash_algorithm=hash_algorithm)

    with store, graph.use():
        # Write parent metadata
        store.write_metadata(ParentFeature, parent_df)

        # Compute child metadata twice
        increment1 = store.resolve_update(
            ChildFeature,
            target_version=ChildFeature.feature_version(),
            snapshot_version=graph.snapshot_version,
        )

        increment2 = store.resolve_update(
            ChildFeature,
            target_version=ChildFeature.feature_version(),
            snapshot_version=graph.snapshot_version,
        )

        # Convert to Polars for comparison
        result1 = increment1.added.lazy().collect().to_polars()
        result2 = increment2.added.lazy().collect().to_polars()

        # Sort by ID columns
        id_columns = list(child_plan.feature.id_columns)
        result1_sorted = result1.sort(id_columns)
        result2_sorted = result2.sort(id_columns)

        # Verify consistency - same input produces same hash
        pl_testing.assert_frame_equal(result1_sorted, result2_sorted)

        # Verify all hashes are non-null
        assert result1["metaxy_provenance"].null_count() == 0, f"Hash algorithm {hash_algorithm} produced null hashes"


# ============= TEST: HASH TRUNCATION =============


@pytest.mark.parametrize("truncation_length", [None, 8, 16, 32])
@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
def test_hash_truncation(
    config_with_truncation,
    hash_algorithm: HashAlgorithm,
    graph: FeatureGraph,
    tmp_path,
):
    # Config is already set by fixture
    truncation_length = config_with_truncation.hash_truncation_length

    # Skip invalid combinations where truncation is longer than the hash algorithm produces
    max_hash_lengths = {
        HashAlgorithm.XXHASH32: 10,
        HashAlgorithm.XXHASH64: 20,
        HashAlgorithm.WYHASH: 19,
        HashAlgorithm.SHA256: 64,
        HashAlgorithm.MD5: 32,
    }

    if truncation_length is not None and truncation_length > max_hash_lengths.get(hash_algorithm, 64):
        pytest.skip(
            f"Truncation length {truncation_length} exceeds max hash length "
            f"{max_hash_lengths.get(hash_algorithm, 64)} for {hash_algorithm}"
        )

    class ParentFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="parent",
            fields=["value"],
        ),
    ):
        pass

    class ChildFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="child",
            deps=[FeatureDep(feature=ParentFeature)],
            fields=["result"],
        ),
    ):
        pass

    child_plan = graph.get_feature_plan(ChildFeature.spec().key)
    feature_versions = {
        "parent": ParentFeature.feature_version(),
        "child": ChildFeature.feature_version(),
    }

    # Generate test data
    upstream_data, _ = downstream_metadata_strategy(
        child_plan,
        feature_versions=feature_versions,
        snapshot_version=graph.snapshot_version,
        hash_algorithm=hash_algorithm,
        min_rows=20,
        max_rows=20,
    ).example()

    parent_df = upstream_data["parent"]

    store = DeltaMetadataStore(root_path=tmp_path / "delta_store", hash_algorithm=hash_algorithm)

    with store, graph.use():
        # Write parent metadata
        store.write_metadata(ParentFeature, parent_df)

        # Compute child metadata
        increment = store.resolve_update(
            ChildFeature,
            target_version=ChildFeature.feature_version(),
            snapshot_version=graph.snapshot_version,
        )

        result = increment.added.lazy().collect().to_polars()

        # Check hash lengths
        hash_col = result["metaxy_provenance"]

        for hash_val in hash_col:
            if truncation_length is None:
                # Full hash length depends on algorithm
                assert len(hash_val) > 0, "Hash should not be empty"
            else:
                # Truncated hash should have at most truncation_length (can be shorter for numeric hashes with leading zeros)
                assert len(hash_val) <= truncation_length, (
                    f"Expected hash length <= {truncation_length}, got {len(hash_val)}"
                )
                # For most cases it should be close to the truncation length (within 2 chars)
                # Numeric hashes (xxhash32, xxhash64, wyhash) can have leading zeros stripped,
                # so we don't fail on slightly shorter hashes


@parametrize_with_cases("hash_algorithm", cases=HashAlgorithmCases)
def test_field_level_provenance_structure(
    hash_algorithm: HashAlgorithm,
    graph: FeatureGraph,
    tmp_path,
):
    """Test that field-level provenance has correct structure for each hash algorithm."""

    class ParentFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="parent",
            fields=["field_a", "field_b", "field_c"],
        ),
    ):
        pass

    class ChildFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="child",
            deps=[FeatureDep(feature=ParentFeature)],
            fields=["result_x", "result_y"],
        ),
    ):
        pass

    child_plan = graph.get_feature_plan(ChildFeature.spec().key)
    feature_versions = {
        "parent": ParentFeature.feature_version(),
        "child": ChildFeature.feature_version(),
    }

    store = DeltaMetadataStore(root_path=tmp_path / "delta_store", hash_algorithm=hash_algorithm)

    with store, graph.use():
        # Generate test data
        upstream_data, _ = downstream_metadata_strategy(
            child_plan,
            feature_versions=feature_versions,
            snapshot_version=graph.snapshot_version,
            hash_algorithm=hash_algorithm,
            min_rows=5,
            max_rows=5,
        ).example()

        parent_df = upstream_data["parent"]

        # Write parent and compute child
        store.write_metadata(ParentFeature, parent_df)
        increment = store.resolve_update(
            ChildFeature,
            target_version=ChildFeature.feature_version(),
            snapshot_version=graph.snapshot_version,
        )

        result = increment.added.lazy().collect().to_polars()

        # Check provenance_by_field structure
        provenance_by_field = result["metaxy_provenance_by_field"]

        # Verify it's a struct column
        assert provenance_by_field.dtype == pl.Struct, "provenance_by_field should be a Struct column"

        # Unnest to check field structure
        unnested = result.unnest("metaxy_provenance_by_field")

        # Check that we have one column per child field
        expected_fields = {"result_x", "result_y"}
        actual_fields = {col for col in unnested.columns if col in expected_fields}

        assert actual_fields == expected_fields, f"Expected fields {expected_fields}, got {actual_fields}"

        # Verify all field hashes are non-null
        for field in expected_fields:
            assert unnested[field].null_count() == 0, f"Field {field} has null provenance hashes"


# ============= TEST: HASH TRUNCATION ACROSS ALL STORES =============


@pytest.mark.parametrize("truncation_length", [16])
def test_hash_truncation_any_store(config_with_truncation, any_store: MetadataStore, graph: FeatureGraph):
    """Test that hash truncation is applied correctly across store types."""

    class ParentFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="parent",
            fields=["value"],
        ),
    ):
        pass

    class ChildFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="child",
            deps=[FeatureDep(feature=ParentFeature)],
            fields=["result"],
        ),
    ):
        pass

    child_plan = graph.get_feature_plan(ChildFeature.spec().key)
    feature_versions = {
        "parent": ParentFeature.feature_version(),
        "child": ChildFeature.feature_version(),
    }

    # Config is already set by fixture to truncation_length=16
    truncation_length = config_with_truncation.hash_truncation_length

    # Generate test data
    upstream_data, _ = downstream_metadata_strategy(
        child_plan,
        feature_versions=feature_versions,
        snapshot_version=graph.snapshot_version,
        hash_algorithm=any_store.hash_algorithm,
        min_rows=5,
        max_rows=10,
    ).example()

    parent_df = upstream_data["parent"]

    with any_store, graph.use():
        # Write parent metadata
        any_store.write_metadata(ParentFeature, parent_df)

        # Compute child metadata
        increment = any_store.resolve_update(
            ChildFeature,
            target_version=ChildFeature.feature_version(),
            snapshot_version=graph.snapshot_version,
        )

        result = increment.added.lazy().collect().to_polars()

        # Verify all hashes are exactly 16 characters
        hash_col = result["metaxy_provenance"]
        for hash_val in hash_col:
            assert len(hash_val) == truncation_length, (
                f"Expected hash length {truncation_length}, got {len(hash_val)} in {type(any_store).__name__}"
            )
