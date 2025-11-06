"""Integration tests for ProvenanceTracker with FeatureGraph.

These tests verify that trackers integrate correctly with the broader
Metaxy system including FeatureGraph, FeatureDep renames, and filters.
"""

from __future__ import annotations

from typing import cast

import narwhals as nw
import polars as pl
import pytest
from metaxy.data_versioning import HashAlgorithm

from metaxy.models.feature import FeatureGraph, TestingFeature
from metaxy.models.feature_spec import FeatureDep, SampleFeatureSpec
from metaxy.models.field import FieldDep, FieldSpec
from metaxy.models.types import FeatureKey, FieldKey
from metaxy.provenance.polars import PolarsProvenanceTracker


def test_feature_dep_renames(graph: FeatureGraph, snapshot) -> None:
    """Test that FeatureDep renames are correctly applied."""

    class UpstreamFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream"]),
            fields=[
                FieldSpec(key=FieldKey(["original_name"]), code_version="1"),
            ],
        ),
    ):
        pass

    class DownstreamFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["downstream"]),
            deps=[
                FeatureDep(
                    feature=FeatureKey(["upstream"]),
                    rename={"original_name": "renamed_field"},
                )
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["default"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["upstream"]),
                            fields=[FieldKey(["original_name"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    plan = graph.get_feature_plan(DownstreamFeature.spec().key)
    tracker = PolarsProvenanceTracker(plan)

    # Create upstream metadata with original name
    upstream_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "original_name": ["value1", "value2"],
                "metaxy_provenance_by_field": [
                    {"original_name": "hash1"},
                    {"original_name": "hash2"},
                ],
                "metaxy_provenance": ["prov1", "prov2"],
            }
        ).lazy()
    )

    upstream = {FeatureKey(["upstream"]): upstream_metadata}

    # Prepare upstream - should apply rename
    prepared = tracker.prepare_upstream(
        upstream=upstream,
        filters={},
    )

    prepared_df = prepared.collect()

    # Verify rename was applied
    assert "renamed_field" in prepared_df.columns
    assert "original_name" not in prepared_df.columns

    # Verify values were preserved
    assert prepared_df["renamed_field"].to_list() == ["value1", "value2"]

    # Compute full provenance
    result = tracker.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        hash_length=32,
        filters={},
    )

    result_df = result.collect()

    # Snapshot the provenance
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


def test_feature_dep_column_selection(graph: FeatureGraph) -> None:
    """Test that FeatureDep column selection works correctly."""

    class UpstreamFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream"]),
            fields=[
                FieldSpec(key=FieldKey(["field_a"]), code_version="1"),
                FieldSpec(key=FieldKey(["field_b"]), code_version="1"),
                FieldSpec(key=FieldKey(["field_c"]), code_version="1"),
            ],
        ),
    ):
        pass

    class DownstreamFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["downstream"]),
            deps=[
                FeatureDep(
                    feature=FeatureKey(["upstream"]),
                    columns=("field_a", "field_c"),  # Only select specific columns
                )
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["default"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["upstream"]),
                            fields=[FieldKey(["field_a"]), FieldKey(["field_c"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    plan = graph.get_feature_plan(DownstreamFeature.spec().key)
    tracker = PolarsProvenanceTracker(plan)

    # Create upstream metadata with all fields
    upstream_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "field_a": ["a1", "a2"],
                "field_b": ["b1", "b2"],
                "field_c": ["c1", "c2"],
                "metaxy_provenance_by_field": [
                    {"field_a": "hash_a1", "field_b": "hash_b1", "field_c": "hash_c1"},
                    {"field_a": "hash_a2", "field_b": "hash_b2", "field_c": "hash_c2"},
                ],
                "metaxy_provenance": ["prov1", "prov2"],
            }
        ).lazy()
    )

    upstream = {FeatureKey(["upstream"]): upstream_metadata}

    # Prepare upstream - should only include selected columns
    prepared = tracker.prepare_upstream(
        upstream=upstream,
        filters={},
    )

    prepared_df = prepared.collect()

    # Verify only selected columns are present (plus provenance columns)
    assert "field_a" in prepared_df.columns
    assert "field_c" in prepared_df.columns
    assert "field_b" not in prepared_df.columns  # Should be excluded


def test_multi_upstream_join_on_common_id_columns(graph: FeatureGraph) -> None:
    """Test that multiple upstreams are joined on common ID columns."""

    class UpstreamA(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream_a"]),
            fields=[FieldSpec(key=FieldKey(["data_a"]), code_version="1")],
        ),
    ):
        pass

    class UpstreamB(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream_b"]),
            fields=[FieldSpec(key=FieldKey(["data_b"]), code_version="1")],
        ),
    ):
        pass

    class DownstreamFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["downstream"]),
            deps=[
                FeatureDep(feature=FeatureKey(["upstream_a"])),
                FeatureDep(feature=FeatureKey(["upstream_b"])),
            ],
            fields=[
                FieldSpec(key=FieldKey(["default"]), code_version="1"),
            ],
        ),
    ):
        pass

    plan = graph.get_feature_plan(DownstreamFeature.spec().key)
    tracker = PolarsProvenanceTracker(plan)

    # Create upstream metadata with partial overlap on sample_uid
    upstream_a_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],  # Has 1, 2, 3
                "data_a": ["a1", "a2", "a3"],
                "metaxy_provenance_by_field": [
                    {"data_a": "hash_a1"},
                    {"data_a": "hash_a2"},
                    {"data_a": "hash_a3"},
                ],
                "metaxy_provenance": ["prov_a1", "prov_a2", "prov_a3"],
            }
        ).lazy()
    )

    upstream_b_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [2, 3, 4],  # Has 2, 3, 4
                "data_b": ["b2", "b3", "b4"],
                "metaxy_provenance_by_field": [
                    {"data_b": "hash_b2"},
                    {"data_b": "hash_b3"},
                    {"data_b": "hash_b4"},
                ],
                "metaxy_provenance": ["prov_b2", "prov_b3", "prov_b4"],
            }
        ).lazy()
    )

    upstream = {
        FeatureKey(["upstream_a"]): upstream_a_metadata,
        FeatureKey(["upstream_b"]): upstream_b_metadata,
    }

    # Prepare upstream - should inner join on sample_uid
    prepared = tracker.prepare_upstream(
        upstream=upstream,
        filters={},
    )

    prepared_df = prepared.collect()

    # Only samples 2 and 3 should be present (intersection)
    assert len(prepared_df) == 2
    assert set(prepared_df["sample_uid"].to_list()) == {2, 3}

    # Both data columns should be present
    assert "data_a" in prepared_df.columns
    assert "data_b" in prepared_df.columns


def test_filters_applied_before_join(graph: FeatureGraph) -> None:
    """Test that filters are applied to upstream before joining."""

    class UpstreamA(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream_a"]),
            fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
        ),
    ):
        pass

    class UpstreamB(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream_b"]),
            fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
        ),
    ):
        pass

    class DownstreamFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["downstream"]),
            deps=[
                FeatureDep(
                    feature=FeatureKey(["upstream_a"]),
                    rename={"value": "value_a"},  # Add rename to avoid collision
                ),
                FeatureDep(
                    feature=FeatureKey(["upstream_b"]),
                    rename={"value": "value_b"},  # Add rename to avoid collision
                ),
            ],
            fields=[
                FieldSpec(key=FieldKey(["default"]), code_version="1"),
            ],
        ),
    ):
        pass

    plan = graph.get_feature_plan(DownstreamFeature.spec().key)
    tracker = PolarsProvenanceTracker(plan)

    upstream_a_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3, 4, 5],
                "value": [10, 20, 30, 40, 50],
                "metaxy_provenance_by_field": [
                    {"value": f"hash_{i}"} for i in range(1, 6)
                ],
                "metaxy_provenance": [f"prov_{i}" for i in range(1, 6)],
            }
        ).lazy()
    )

    upstream_b_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3, 4, 5],
                "value": [15, 25, 35, 45, 55],
                "metaxy_provenance_by_field": [
                    {"value": f"hash_{i}"} for i in range(1, 6)
                ],
                "metaxy_provenance": [f"prov_{i}" for i in range(1, 6)],
            }
        ).lazy()
    )

    upstream = {
        FeatureKey(["upstream_a"]): upstream_a_metadata,
        FeatureKey(["upstream_b"]): upstream_b_metadata,
    }

    # Apply filters: only include samples where sample_uid > 2 from upstream_a
    # and sample_uid < 5 from upstream_b
    filters = {
        FeatureKey(["upstream_a"]): [nw.col("sample_uid") > 2],
        FeatureKey(["upstream_b"]): [nw.col("sample_uid") < 5],
    }

    prepared = tracker.prepare_upstream(
        upstream=upstream,
        filters=filters,
    )

    prepared_df = cast(pl.LazyFrame, prepared.to_native()).collect()

    # After filtering: upstream_a has [3, 4, 5], upstream_b has [1, 2, 3, 4]
    # After join: should have [3, 4] (intersection)
    assert len(prepared_df) == 2
    assert set(prepared_df["sample_uid"].to_list()) == {3, 4}


@pytest.mark.skip(reason="TODO: Fix column collision issue in multi-level dependencies")
def test_complex_dependency_graph(graph: FeatureGraph, snapshot) -> None:
    """Test tracker with a complex multi-level dependency graph."""
    # Note: This test is skipped but needs a valid plan for type checking

    # Level 0: Root features (separate features to avoid collisions)
    class RawA(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["raw_a"]),
            fields=[
                FieldSpec(key=FieldKey(["sensor"]), code_version="1"),
            ],
        ),
    ):
        pass

    class RawB(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["raw_b"]),
            fields=[
                FieldSpec(key=FieldKey(["sensor"]), code_version="1"),
            ],
        ),
    ):
        pass

    # Level 1: Process raw data
    class ProcessedA(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["processed_a"]),
            deps=[FeatureDep(feature=FeatureKey(["raw_a"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["result"]),
                    code_version="2",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["raw_a"]),
                            fields=[FieldKey(["sensor"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    class ProcessedB(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["processed_b"]),
            deps=[FeatureDep(feature=FeatureKey(["raw_b"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["result"]),
                    code_version="3",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["raw_b"]),
                            fields=[FieldKey(["sensor"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    # Level 2: Combine processed data
    class FinalAnalysis(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["analysis"]),
            deps=[
                FeatureDep(
                    feature=FeatureKey(["processed_a"]),
                    rename={"sensor": "sensor_a"},  # Rename to avoid collision
                ),
                FeatureDep(
                    feature=FeatureKey(["processed_b"]),
                    rename={"sensor": "sensor_b"},  # Rename to avoid collision
                ),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["combined"]),
                    code_version="10",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["processed_a"]),
                            fields=[FieldKey(["result"])],
                        ),
                        FieldDep(
                            feature=FeatureKey(["processed_b"]),
                            fields=[FieldKey(["result"])],
                        ),
                    ],
                ),
            ],
        ),
    ):
        pass

    # Create metadata for raw data (separate features)
    raw_a_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "sensor": [100, 200],
                "metaxy_provenance_by_field": [
                    {"sensor": "raw_a1"},
                    {"sensor": "raw_a2"},
                ],
                "metaxy_provenance": ["raw_prov_a1", "raw_prov_a2"],
            }
        ).lazy()
    )

    raw_b_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "sensor": [150, 250],
                "metaxy_provenance_by_field": [
                    {"sensor": "raw_b1"},
                    {"sensor": "raw_b2"},
                ],
                "metaxy_provenance": ["raw_prov_b1", "raw_prov_b2"],
            }
        ).lazy()
    )

    # Compute provenance for ProcessedA
    processed_a_plan = graph.get_feature_plan(ProcessedA.spec().key)
    processed_a_tracker = PolarsProvenanceTracker(processed_a_plan)
    processed_a_result = processed_a_tracker.load_upstream_with_provenance(
        upstream={FeatureKey(["raw_a"]): raw_a_metadata},
        hash_algo=HashAlgorithm.XXHASH64,
        hash_length=32,
        filters={},
    )

    # Compute provenance for ProcessedB
    processed_b_plan = graph.get_feature_plan(ProcessedB.spec().key)
    processed_b_tracker = PolarsProvenanceTracker(processed_b_plan)
    processed_b_result = processed_b_tracker.load_upstream_with_provenance(
        upstream={FeatureKey(["raw_b"]): raw_b_metadata},
        hash_algo=HashAlgorithm.XXHASH64,
        hash_length=32,
        filters={},
    )

    # Compute provenance for FinalAnalysis
    final_analysis_plan = graph.get_feature_plan(FinalAnalysis.spec().key)
    final_analysis_tracker = PolarsProvenanceTracker(final_analysis_plan)
    final_result = final_analysis_tracker.load_upstream_with_provenance(
        upstream={
            FeatureKey(["processed_a"]): processed_a_result,
            FeatureKey(["processed_b"]): processed_b_result,
        },
        hash_algo=HashAlgorithm.XXHASH64,
        hash_length=32,
        filters={},
    )

    final_df = final_result.collect()

    # Snapshot the final provenance
    provenance_data = []
    for i in range(len(final_df)):
        provenance_data.append(
            {
                "sample_uid": final_df["sample_uid"][i],
                "field_provenance": final_df["metaxy_provenance_by_field"][i],
                "sample_provenance": final_df["metaxy_provenance"][i],
            }
        )

    assert provenance_data == snapshot


def test_validate_no_colliding_columns(graph: FeatureGraph) -> None:
    """Test that validation catches colliding column names after renames."""

    class UpstreamA(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream_a"]),
            fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
        ),
    ):
        pass

    class UpstreamB(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream_b"]),
            fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
        ),
    ):
        pass

    class BadDownstream(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["bad_downstream"]),
            deps=[
                FeatureDep(feature=FeatureKey(["upstream_a"])),
                FeatureDep(feature=FeatureKey(["upstream_b"])),
                # Missing renames - both upstreams have "data" column
            ],
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    plan = graph.get_feature_plan(BadDownstream.spec().key)
    tracker = PolarsProvenanceTracker(plan)

    upstream_a_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1],
                "data": ["a1"],
                "metaxy_provenance_by_field": [{"data": "hash_a"}],
                "metaxy_provenance": ["prov_a"],
            }
        ).lazy()
    )

    upstream_b_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1],
                "data": ["b1"],
                "metaxy_provenance_by_field": [{"data": "hash_b"}],
                "metaxy_provenance": ["prov_b"],
            }
        ).lazy()
    )

    upstream = {
        FeatureKey(["upstream_a"]): upstream_a_metadata,
        FeatureKey(["upstream_b"]): upstream_b_metadata,
    }

    # Should raise error about colliding columns
    with pytest.raises(ValueError, match="additional shared columns"):
        tracker.prepare_upstream(
            upstream=upstream,
            filters={},
        )


def test_feature_graph_integration_with_provenance_by_field(
    graph: FeatureGraph, snapshot
) -> None:
    """Test integration with FeatureGraph's provenance_by_field() method."""
    # This test verifies that trackers work correctly with the Feature.provenance_by_field()
    # class method that returns the expected field provenance structure

    class MyFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["my_feature"]),
            fields=[
                FieldSpec(key=FieldKey(["field_a"]), code_version="100"),
                FieldSpec(key=FieldKey(["field_b"]), code_version="200"),
            ],
        ),
    ):
        pass

    # Get provenance_by_field from the class method
    provenance_dict = MyFeature.provenance_by_field()

    # Should have entries for both fields
    assert "field_a" in provenance_dict
    assert "field_b" in provenance_dict

    # Different code versions should produce different provenance values
    assert provenance_dict["field_a"] != provenance_dict["field_b"]

    # Snapshot for stability
    assert provenance_dict == snapshot
