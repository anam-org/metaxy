"""Tests for user-defined data_version_by_field functionality.

These tests verify that downstream features correctly compute their provenance
based on upstream data_version_by_field (not provenance_by_field), enabling
users to control version propagation through custom data_version overrides.
"""

from __future__ import annotations

import narwhals as nw
import polars as pl

from metaxy.models.feature import FeatureGraph, TestingFeature
from metaxy.models.feature_spec import FeatureDep, SampleFeatureSpec
from metaxy.models.field import FieldDep, FieldSpec, SpecialFieldDep
from metaxy.models.types import FeatureKey, FieldKey
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm


def test_basic_data_version_override(graph: FeatureGraph, snapshot) -> None:
    """Test basic case: user provides custom data_version_by_field, verify downstream uses it."""

    class ParentFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["parent"]),
            fields=[
                FieldSpec(key=FieldKey(["value"]), code_version="1"),
            ],
        ),
    ):
        pass

    class ChildFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["child"]),
            deps=[FeatureDep(feature=FeatureKey(["parent"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["computed"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["parent"]),
                            fields=[FieldKey(["value"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    # Create parent metadata with custom data_version
    # data_version differs from provenance to test that data_version is used
    parent_df = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "value": ["a", "b"],
                "metaxy_provenance_by_field": [
                    {"value": "provenance_hash_1"},
                    {"value": "provenance_hash_2"},
                ],
                "metaxy_provenance": ["prov_1", "prov_2"],
                "metaxy_data_version_by_field": [
                    {"value": "custom_version_1"},
                    {"value": "custom_version_2"},
                ],
            }
        ).lazy()
    )

    # Get child plan and engine
    child_plan = graph.get_feature_plan(ChildFeature.spec().key)
    engine = PolarsVersioningEngine(child_plan)

    # Compute child provenance from parent
    upstream = {FeatureKey(["parent"]): parent_df}
    result = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect().to_polars()

    # Extract the computed field provenance for the child's "computed" field
    # This should be based on parent's data_version, NOT provenance
    computed_field_provenance = [
        row["metaxy_provenance_by_field"]["computed"]
        for row in result_df.iter_rows(named=True)
    ]

    # The provenance should be computed from custom_version_1/2, not provenance_hash_1/2
    # We can't predict exact hash values, but we can verify:
    # 1. The two samples have different field provenances (since data_versions differ)
    assert computed_field_provenance[0] != computed_field_provenance[1]
    # 2. Both are non-empty strings
    assert len(computed_field_provenance[0]) > 0
    assert len(computed_field_provenance[1]) > 0

    # Snapshot the provenance and data_version values
    provenance_data = []
    for i in range(len(result_df)):
        provenance_data.append(
            {
                "sample_uid": result_df["sample_uid"][i],
                "field_provenance": result_df["metaxy_provenance_by_field"][i],
                "field_data_version": result_df["metaxy_data_version_by_field"][i],
            }
        )
    assert provenance_data == snapshot


def test_propagation_chain_with_data_version(graph: FeatureGraph, snapshot) -> None:
    """Test A→B→C where A has custom data_version, verify C's provenance traces back correctly."""

    class FeatureA(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["A"]),
            fields=[
                FieldSpec(key=FieldKey(["field_a"]), code_version="1"),
            ],
        ),
    ):
        pass

    class FeatureB(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["B"]),
            deps=[FeatureDep(feature=FeatureKey(["A"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["field_b"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["A"]),
                            fields=[FieldKey(["field_a"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    class FeatureC(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["C"]),
            deps=[FeatureDep(feature=FeatureKey(["B"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["field_c"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["B"]),
                            fields=[FieldKey(["field_b"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    # Feature A: Root with custom data_version
    a_df = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "field_a": ["x", "y"],
                "metaxy_provenance_by_field": [
                    {"field_a": "a_prov_1"},
                    {"field_a": "a_prov_2"},
                ],
                "metaxy_provenance": ["a_sample_prov_1", "a_sample_prov_2"],
                "metaxy_data_version_by_field": [
                    {"field_a": "a_version_1"},
                    {"field_a": "a_version_1"},  # Same version for both samples!
                ],
            }
        ).lazy()
    )

    # Compute B from A
    b_plan = graph.get_feature_plan(FeatureB.spec().key)
    b_engine = PolarsVersioningEngine(b_plan)
    b_result = b_engine.load_upstream_with_provenance(
        upstream={FeatureKey(["A"]): a_df},
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    # Compute C from B (keep as lazy frame)
    c_plan = graph.get_feature_plan(FeatureC.spec().key)
    c_engine = PolarsVersioningEngine(c_plan)
    c_result = c_engine.load_upstream_with_provenance(
        upstream={FeatureKey(["B"]): b_result},
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    # Extract field provenances
    b_field_provs = (
        b_result.collect().to_polars()["metaxy_provenance_by_field"].to_list()
    )
    c_field_provs = (
        c_result.collect().to_polars()["metaxy_provenance_by_field"].to_list()
    )

    # B should have same field provenance for both samples
    # (because A's data_version is the same for both)
    assert b_field_provs[0]["field_b"] == b_field_provs[1]["field_b"]

    # C should also have same field provenance for both samples
    # (because B's data_version is the same, which came from A's data_version)
    assert c_field_provs[0]["field_c"] == c_field_provs[1]["field_c"]

    # Snapshot the provenance chain for B and C
    b_df = b_result.collect().to_polars()
    c_df = c_result.collect().to_polars()

    chain_data = {
        "feature_b": [
            {
                "sample_uid": b_df["sample_uid"][i],
                "field_provenance": b_df["metaxy_provenance_by_field"][i],
                "field_data_version": b_df["metaxy_data_version_by_field"][i],
            }
            for i in range(len(b_df))
        ],
        "feature_c": [
            {
                "sample_uid": c_df["sample_uid"][i],
                "field_provenance": c_df["metaxy_provenance_by_field"][i],
                "field_data_version": c_df["metaxy_data_version_by_field"][i],
            }
            for i in range(len(c_df))
        ],
    }
    assert chain_data == snapshot


def test_selective_field_override(graph: FeatureGraph, snapshot) -> None:
    """Test overriding only some fields in data_version_by_field."""

    class ParentFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["parent"]),
            fields=[
                FieldSpec(key=FieldKey(["field1"]), code_version="1"),
                FieldSpec(key=FieldKey(["field2"]), code_version="1"),
            ],
        ),
    ):
        pass

    class ChildFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["child"]),
            deps=[FeatureDep(feature=FeatureKey(["parent"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["uses_field1"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["parent"]),
                            fields=[FieldKey(["field1"])],
                        )
                    ],
                ),
                FieldSpec(
                    key=FieldKey(["uses_field2"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["parent"]),
                            fields=[FieldKey(["field2"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    # Parent: field1 has custom data_version, field2 uses provenance (data_version = provenance)
    parent_df = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "field1": ["a", "b"],
                "field2": ["c", "d"],
                "metaxy_provenance_by_field": [
                    {"field1": "field1_prov_1", "field2": "field2_prov_1"},
                    {"field1": "field1_prov_2", "field2": "field2_prov_2"},
                ],
                "metaxy_provenance": ["parent_prov_1", "parent_prov_2"],
                "metaxy_data_version_by_field": [
                    {
                        "field1": "field1_custom",  # Custom version (same for both)
                        "field2": "field2_prov_1",  # Defaults to provenance
                    },
                    {
                        "field1": "field1_custom",  # Custom version (same for both)
                        "field2": "field2_prov_2",  # Defaults to provenance
                    },
                ],
            }
        ).lazy()
    )

    child_plan = graph.get_feature_plan(ChildFeature.spec().key)
    engine = PolarsVersioningEngine(child_plan)

    result = engine.load_upstream_with_provenance(
        upstream={FeatureKey(["parent"]): parent_df},
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect().to_polars()

    # Extract field provenances
    field_provs = result_df["metaxy_provenance_by_field"].to_list()

    # uses_field1 should have SAME provenance for both samples (custom version is same)
    assert field_provs[0]["uses_field1"] == field_provs[1]["uses_field1"]

    # uses_field2 should have DIFFERENT provenance for both samples (uses default provenance)
    assert field_provs[0]["uses_field2"] != field_provs[1]["uses_field2"]

    # Snapshot the selective override results
    provenance_data = []
    for i in range(len(result_df)):
        provenance_data.append(
            {
                "sample_uid": result_df["sample_uid"][i],
                "field_provenance": result_df["metaxy_provenance_by_field"][i],
                "field_data_version": result_df["metaxy_data_version_by_field"][i],
            }
        )
    assert provenance_data == snapshot


def test_default_behavior_no_override(graph: FeatureGraph, snapshot) -> None:
    """Test that when no override is provided, data_version equals provenance."""

    class ParentFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["parent"]),
            fields=[
                FieldSpec(key=FieldKey(["value"]), code_version="1"),
            ],
        ),
    ):
        pass

    class ChildFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["child"]),
            deps=[FeatureDep(feature=FeatureKey(["parent"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["computed"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["parent"]),
                            fields=[FieldKey(["value"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    # Parent metadata where data_version equals provenance (default behavior)
    parent_df = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "value": ["a", "b"],
                "metaxy_provenance_by_field": [
                    {"value": "hash_1"},
                    {"value": "hash_2"},
                ],
                "metaxy_provenance": ["prov_1", "prov_2"],
                "metaxy_data_version_by_field": [
                    {"value": "hash_1"},  # Same as provenance
                    {"value": "hash_2"},  # Same as provenance
                ],
            }
        ).lazy()
    )

    child_plan = graph.get_feature_plan(ChildFeature.spec().key)
    engine = PolarsVersioningEngine(child_plan)

    result = engine.load_upstream_with_provenance(
        upstream={FeatureKey(["parent"]): parent_df},
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect().to_polars()

    # Extract computed field provenance
    computed_provs = [
        row["metaxy_provenance_by_field"]["computed"]
        for row in result_df.iter_rows(named=True)
    ]

    # Should have different provenances (since parent's data_versions differ)
    assert computed_provs[0] != computed_provs[1]
    assert len(computed_provs[0]) > 0
    assert len(computed_provs[1]) > 0

    # Snapshot the default behavior results
    provenance_data = []
    for i in range(len(result_df)):
        provenance_data.append(
            {
                "sample_uid": result_df["sample_uid"][i],
                "field_provenance": result_df["metaxy_provenance_by_field"][i],
                "field_data_version": result_df["metaxy_data_version_by_field"][i],
            }
        )
    assert provenance_data == snapshot


def test_multiple_upstreams_with_overrides(graph: FeatureGraph, snapshot) -> None:
    """Test feature with multiple parents where some have data_version overrides."""

    class Parent1(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["parent1"]),
            fields=[
                FieldSpec(key=FieldKey(["field_p1"]), code_version="1"),
            ],
        ),
    ):
        pass

    class Parent2(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["parent2"]),
            fields=[
                FieldSpec(key=FieldKey(["field_p2"]), code_version="1"),
            ],
        ),
    ):
        pass

    class ChildFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["child"]),
            deps=[
                FeatureDep(feature=FeatureKey(["parent1"])),
                FeatureDep(feature=FeatureKey(["parent2"])),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["fusion"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                ),
            ],
        ),
    ):
        pass

    # Parent1: Custom data_version (same for both samples)
    p1_df = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "field_p1": ["a", "b"],
                "metaxy_provenance_by_field": [
                    {"field_p1": "p1_prov_1"},
                    {"field_p1": "p1_prov_2"},
                ],
                "metaxy_provenance": ["p1_sample_1", "p1_sample_2"],
                "metaxy_data_version_by_field": [
                    {"field_p1": "p1_custom"},  # Same custom version
                    {"field_p1": "p1_custom"},  # Same custom version
                ],
            }
        ).lazy()
    )

    # Parent2: Default behavior (data_version = provenance, different for samples)
    p2_df = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "field_p2": ["c", "d"],
                "metaxy_provenance_by_field": [
                    {"field_p2": "p2_prov_1"},
                    {"field_p2": "p2_prov_2"},
                ],
                "metaxy_provenance": ["p2_sample_1", "p2_sample_2"],
                "metaxy_data_version_by_field": [
                    {"field_p2": "p2_prov_1"},  # Same as provenance
                    {"field_p2": "p2_prov_2"},  # Same as provenance
                ],
            }
        ).lazy()
    )

    child_plan = graph.get_feature_plan(ChildFeature.spec().key)
    engine = PolarsVersioningEngine(child_plan)

    result = engine.load_upstream_with_provenance(
        upstream={
            FeatureKey(["parent1"]): p1_df,
            FeatureKey(["parent2"]): p2_df,
        },
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect().to_polars()

    # Extract fusion field provenance
    fusion_provs = [
        row["metaxy_provenance_by_field"]["fusion"]
        for row in result_df.iter_rows(named=True)
    ]

    # Since parent1 has same data_version but parent2 differs,
    # the fusion field should have different provenances for the two samples
    assert fusion_provs[0] != fusion_provs[1]

    # Snapshot the multi-upstream fusion results
    provenance_data = []
    for i in range(len(result_df)):
        provenance_data.append(
            {
                "sample_uid": result_df["sample_uid"][i],
                "field_provenance": result_df["metaxy_provenance_by_field"][i],
                "field_data_version": result_df["metaxy_data_version_by_field"][i],
            }
        )
    assert provenance_data == snapshot


def test_data_version_propagation_with_renames(graph: FeatureGraph, snapshot) -> None:
    """Test that data_version propagation works correctly with FeatureDep renames."""

    class ParentFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["parent"]),
            fields=[
                FieldSpec(key=FieldKey(["original_name"]), code_version="1"),
            ],
        ),
    ):
        pass

    class ChildFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["child"]),
            deps=[
                FeatureDep(
                    feature=FeatureKey(["parent"]),
                    rename={"original_name": "renamed_field"},
                )
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["computed"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["parent"]),
                            fields=[FieldKey(["original_name"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    # Parent with custom data_version
    parent_df = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "original_name": ["a", "b"],
                "metaxy_provenance_by_field": [
                    {"original_name": "prov_1"},
                    {"original_name": "prov_2"},
                ],
                "metaxy_provenance": ["sample_prov_1", "sample_prov_2"],
                "metaxy_data_version_by_field": [
                    {"original_name": "custom_v1"},
                    {"original_name": "custom_v2"},
                ],
            }
        ).lazy()
    )

    child_plan = graph.get_feature_plan(ChildFeature.spec().key)
    engine = PolarsVersioningEngine(child_plan)

    # First verify that renames are applied correctly
    prepared = engine.prepare_upstream(
        upstream={FeatureKey(["parent"]): parent_df},
        filters={},
    )
    prepared_df = prepared.collect()

    # Verify rename was applied to user field
    assert "renamed_field" in prepared_df.columns
    assert "original_name" not in prepared_df.columns

    # Verify system columns are properly renamed
    assert "metaxy_data_version_by_field__parent" in prepared_df.columns

    # Now compute full provenance
    result = engine.load_upstream_with_provenance(
        upstream={FeatureKey(["parent"]): parent_df},
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect().to_polars()

    # Verify child provenance is computed correctly from parent's data_version
    computed_provs = [
        row["metaxy_provenance_by_field"]["computed"]
        for row in result_df.iter_rows(named=True)
    ]

    # Should have different provenances (from custom_v1 and custom_v2)
    assert computed_provs[0] != computed_provs[1]
    assert len(computed_provs[0]) > 0
    assert len(computed_provs[1]) > 0

    # Snapshot the provenance with renames
    provenance_data = []
    for i in range(len(result_df)):
        provenance_data.append(
            {
                "sample_uid": result_df["sample_uid"][i],
                "field_provenance": result_df["metaxy_provenance_by_field"][i],
                "field_data_version": result_df["metaxy_data_version_by_field"][i],
            }
        )
    assert provenance_data == snapshot


def test_data_version_cleanup_in_result(graph: FeatureGraph, snapshot) -> None:
    """Test that renamed upstream data_version columns are cleaned up from result."""

    class ParentFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["parent"]),
            fields=[
                FieldSpec(key=FieldKey(["value"]), code_version="1"),
            ],
        ),
    ):
        pass

    class ChildFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["child"]),
            deps=[FeatureDep(feature=FeatureKey(["parent"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["computed"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["parent"]),
                            fields=[FieldKey(["value"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    parent_df = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "value": ["a", "b"],
                "metaxy_provenance_by_field": [
                    {"value": "prov_1"},
                    {"value": "prov_2"},
                ],
                "metaxy_provenance": ["sample_prov_1", "sample_prov_2"],
                "metaxy_data_version_by_field": [
                    {"value": "version_1"},
                    {"value": "version_2"},
                ],
            }
        ).lazy()
    )

    child_plan = graph.get_feature_plan(ChildFeature.spec().key)
    engine = PolarsVersioningEngine(child_plan)

    result = engine.load_upstream_with_provenance(
        upstream={FeatureKey(["parent"]): parent_df},
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()

    # Verify that renamed upstream system columns are NOT in the result
    result_columns = result_df.columns
    assert "metaxy_data_version_by_field__parent" not in result_columns
    assert "metaxy_provenance_by_field__parent" not in result_columns
    assert "metaxy_provenance__parent" not in result_columns

    # Verify that child's own system columns ARE in the result
    assert "metaxy_provenance_by_field" in result_columns
    assert "metaxy_provenance" in result_columns
    assert "metaxy_data_version_by_field" in result_columns
    assert "metaxy_data_version" in result_columns

    # Snapshot the cleaned result
    result_df_polars = result_df.to_polars()
    provenance_data = []
    for i in range(len(result_df_polars)):
        provenance_data.append(
            {
                "sample_uid": result_df_polars["sample_uid"][i],
                "field_provenance": result_df_polars["metaxy_provenance_by_field"][i],
                "field_data_version": result_df_polars["metaxy_data_version_by_field"][
                    i
                ],
            }
        )
    assert provenance_data == snapshot
