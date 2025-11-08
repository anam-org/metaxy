"""Tests for metadata generation strategies using Hypothesis."""

import polars as pl
import pytest
from hypothesis import given, settings

from metaxy import (
    Feature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    SampleFeatureSpec,
)
from metaxy._testing.parametric import (
    downstream_metadata_strategy,
    feature_metadata_strategy,
    upstream_metadata_strategy,
)
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.models.constants import (
    ALL_SYSTEM_COLUMNS,
    METAXY_FEATURE_VERSION,
    METAXY_PROVENANCE_BY_FIELD,
    METAXY_SNAPSHOT_VERSION,
)


def test_feature_metadata_strategy_basic(graph: FeatureGraph) -> None:
    """Test basic metadata generation for a single feature."""

    class MyFeature(
        Feature,
        spec=SampleFeatureSpec(
            key="my/feature",
            fields=["field1", "field2"],
        ),
    ):
        pass

    spec = MyFeature.spec()
    feature_version = MyFeature.feature_version()
    snapshot_version = "test_snapshot_v1"

    @given(
        feature_metadata_strategy(
            spec,
            feature_version=feature_version,
            snapshot_version=snapshot_version,
            min_rows=5,
            max_rows=10,
        )
    )
    @settings(max_examples=10)
    def property_test(df: pl.DataFrame) -> None:
        # Check row count
        assert 5 <= len(df) <= 10

        # Check ID columns exist
        assert "sample_uid" in df.columns

        # Check all system columns exist
        assert ALL_SYSTEM_COLUMNS.issubset(set(df.columns))

        # Check provenance_by_field structure - extract field names from schema
        provenance_schema = df.schema[METAXY_PROVENANCE_BY_FIELD]
        assert isinstance(provenance_schema, pl.Struct)
        field_names = {field.name for field in provenance_schema.fields}
        assert field_names == {"field1", "field2"}

        # Check version columns have correct constant values
        assert df[METAXY_FEATURE_VERSION].unique().to_list() == [feature_version]
        assert df[METAXY_SNAPSHOT_VERSION].unique().to_list() == [snapshot_version]

        # Check provenance values are populated (strings should NOT be empty)
        for row in df.iter_rows(named=True):
            provenance = row[METAXY_PROVENANCE_BY_FIELD]
            # Check the exact fields from schema
            for field_name in field_names:
                assert field_name in provenance
                assert isinstance(provenance[field_name], str)
                assert len(provenance[field_name]) > 0, (
                    f"Field '{field_name}' should not be empty"
                )

    property_test()


def test_feature_metadata_strategy_with_id_columns_df(graph: FeatureGraph) -> None:
    """Test that id_columns_df parameter properly aligns ID values."""

    class MyFeature(
        Feature,
        spec=SampleFeatureSpec(
            key="my/feature",
            fields=["field1"],
        ),
    ):
        pass

    spec = MyFeature.spec()
    feature_version = MyFeature.feature_version()
    snapshot_version = "test_snapshot_v1"

    # Create a specific ID column DataFrame
    id_df = pl.DataFrame({"sample_uid": [100, 200, 300]})

    @given(
        feature_metadata_strategy(
            spec,
            feature_version=feature_version,
            snapshot_version=snapshot_version,
            id_columns_df=id_df,
        )
    )
    @settings(max_examples=5)
    def property_test(df: pl.DataFrame) -> None:
        # Should use the provided ID values
        assert df["sample_uid"].to_list() == [100, 200, 300]
        assert len(df) == 3

        # All system columns should still be present
        assert ALL_SYSTEM_COLUMNS.issubset(set(df.columns))

    property_test()


def test_upstream_metadata_strategy_single_upstream(graph: FeatureGraph) -> None:
    """Test metadata generation for a plan with one upstream feature."""

    class ParentFeature(
        Feature,
        spec=SampleFeatureSpec(
            key="parent",
            fields=["parent_field"],
        ),
    ):
        pass

    class ChildFeature(
        Feature,
        spec=SampleFeatureSpec(
            key="child",
            deps=[FeatureDep(feature="parent")],
            fields=["child_field"],
        ),
    ):
        pass

    plan = graph.get_feature_plan(FeatureKey(["child"]))

    # Get versions from graph
    feature_versions = {
        "parent": ParentFeature.feature_version(),
    }
    snapshot_version = "test_snapshot_v1"

    @given(
        upstream_metadata_strategy(
            plan,
            feature_versions=feature_versions,
            snapshot_version=snapshot_version,
            min_rows=3,
            max_rows=5,
        )
    )
    @settings(max_examples=10)
    def property_test(upstream_data: dict[str, pl.DataFrame]) -> None:
        # Should have exactly one upstream feature
        assert list(upstream_data.keys()) == ["parent"]

        parent_df = upstream_data["parent"]

        # Check row count
        assert 3 <= len(parent_df) <= 5

        # Check ID columns and all system columns exist
        assert "sample_uid" in parent_df.columns
        assert ALL_SYSTEM_COLUMNS.issubset(set(parent_df.columns))

        # Check provenance structure
        assert parent_df.schema[METAXY_PROVENANCE_BY_FIELD] == pl.Struct(
            [pl.Field("parent_field", pl.String)]
        )

        # Check versions are correct
        assert parent_df[METAXY_FEATURE_VERSION].unique().to_list() == [
            feature_versions["parent"]
        ]
        assert parent_df[METAXY_SNAPSHOT_VERSION].unique().to_list() == [
            snapshot_version
        ]

    property_test()


def test_upstream_metadata_strategy_multiple_upstreams(graph: FeatureGraph) -> None:
    """Test metadata generation for a plan with multiple upstream features."""

    class ParentA(
        Feature,
        spec=SampleFeatureSpec(
            key="parent_a",
            fields=["field_a"],
        ),
    ):
        pass

    class ParentB(
        Feature,
        spec=SampleFeatureSpec(
            key="parent_b",
            fields=["field_b"],
        ),
    ):
        pass

    class ChildFeature(
        Feature,
        spec=SampleFeatureSpec(
            key="child",
            deps=[
                FeatureDep(feature="parent_a"),
                FeatureDep(feature="parent_b"),
            ],
            fields=["result"],
        ),
    ):
        pass

    plan = graph.get_feature_plan(FeatureKey(["child"]))

    # Get versions from graph
    feature_versions = {
        "parent_a": ParentA.feature_version(),
        "parent_b": ParentB.feature_version(),
    }
    snapshot_version = "test_snapshot_v1"

    @given(
        upstream_metadata_strategy(
            plan,
            feature_versions=feature_versions,
            snapshot_version=snapshot_version,
            min_rows=5,
            max_rows=10,
        )
    )
    @settings(max_examples=10)
    def property_test(upstream_data: dict[str, pl.DataFrame]) -> None:
        # Should have both upstream features
        assert set(upstream_data.keys()) == {"parent_a", "parent_b"}

        parent_a_df = upstream_data["parent_a"]
        parent_b_df = upstream_data["parent_b"]

        # Both should have the same number of rows (for joins)
        assert len(parent_a_df) == len(parent_b_df)
        assert 5 <= len(parent_a_df) <= 10

        # Both should have matching sample_uid values (for joins)
        assert (
            parent_a_df["sample_uid"].to_list() == parent_b_df["sample_uid"].to_list()
        )

        # Check provenance structures are different
        assert parent_a_df.schema[METAXY_PROVENANCE_BY_FIELD] == pl.Struct(
            [pl.Field("field_a", pl.String)]
        )
        assert parent_b_df.schema[METAXY_PROVENANCE_BY_FIELD] == pl.Struct(
            [pl.Field("field_b", pl.String)]
        )

        # Check versions
        assert parent_a_df[METAXY_FEATURE_VERSION].unique().to_list() == [
            feature_versions["parent_a"]
        ]
        assert parent_b_df[METAXY_FEATURE_VERSION].unique().to_list() == [
            feature_versions["parent_b"]
        ]

        # Both should have same snapshot version
        assert parent_a_df[METAXY_SNAPSHOT_VERSION].unique().to_list() == [
            snapshot_version
        ]
        assert parent_b_df[METAXY_SNAPSHOT_VERSION].unique().to_list() == [
            snapshot_version
        ]

    property_test()


def test_upstream_metadata_strategy_no_deps(graph: FeatureGraph) -> None:
    """Test that strategy returns empty dict when feature has no dependencies."""

    class RootFeature(
        Feature,
        spec=SampleFeatureSpec(
            key="root",
            fields=["field1"],
        ),
    ):
        pass

    plan = graph.get_feature_plan(FeatureKey(["root"]))

    @given(
        upstream_metadata_strategy(
            plan,
            feature_versions={},
            snapshot_version="test_v1",
        )
    )
    @settings(max_examples=5)
    def property_test(upstream_data: dict[str, pl.DataFrame]) -> None:
        # Should be empty since there are no dependencies
        assert upstream_data == {}

    property_test()


def test_upstream_metadata_strategy_missing_version_error(graph: FeatureGraph) -> None:
    """Test that missing feature versions raise clear errors."""

    class ParentFeature(
        Feature,
        spec=SampleFeatureSpec(
            key="parent",
            fields=["field1"],
        ),
    ):
        pass

    class ChildFeature(
        Feature,
        spec=SampleFeatureSpec(
            key="child",
            deps=[FeatureDep(feature="parent")],
            fields=["result"],
        ),
    ):
        pass

    plan = graph.get_feature_plan(FeatureKey(["child"]))

    # Missing the parent version
    with pytest.raises(ValueError, match="Feature version for 'parent' not found"):
        upstream_metadata_strategy(
            plan,
            feature_versions={},  # Empty!
            snapshot_version="test_v1",
        ).example()


def test_feature_metadata_strategy_exact_rows(graph: FeatureGraph) -> None:
    """Test that num_rows parameter creates exact row count."""

    class MyFeature(
        Feature,
        spec=SampleFeatureSpec(
            key="my/feature",
            fields=["field1"],
        ),
    ):
        pass

    spec = MyFeature.spec()
    feature_version = MyFeature.feature_version()
    snapshot_version = "test_v1"

    @given(
        feature_metadata_strategy(
            spec,
            feature_version=feature_version,
            snapshot_version=snapshot_version,
            num_rows=7,  # Exact count
        )
    )
    @settings(max_examples=5)
    def property_test(df: pl.DataFrame) -> None:
        assert len(df) == 7

    property_test()


def test_downstream_metadata_strategy_single_upstream(graph: FeatureGraph) -> None:
    """Test downstream metadata generation with correct provenance calculation."""

    class ParentFeature(
        Feature,
        spec=SampleFeatureSpec(
            key="parent",
            fields=["parent_field"],
        ),
    ):
        pass

    class ChildFeature(
        Feature,
        spec=SampleFeatureSpec(
            key="child",
            deps=[FeatureDep(feature="parent")],
            fields=["child_field"],
        ),
    ):
        pass

    plan = graph.get_feature_plan(FeatureKey(["child"]))

    # Get versions from graph
    feature_versions = {
        "parent": ParentFeature.feature_version(),
        "child": ChildFeature.feature_version(),
    }
    snapshot_version = "test_snapshot_v1"

    @given(
        downstream_metadata_strategy(
            plan,
            feature_versions=feature_versions,
            snapshot_version=snapshot_version,
            hash_algorithm=HashAlgorithm.XXHASH64,
            min_rows=3,
            max_rows=5,
        )
    )
    @settings(max_examples=10)
    def property_test(data: tuple[dict[str, pl.DataFrame], pl.DataFrame]) -> None:
        upstream_data, downstream_df = data

        # Check upstream data structure
        assert "parent" in upstream_data
        parent_df = upstream_data["parent"]
        assert 3 <= len(parent_df) <= 5

        # Check downstream data structure
        assert len(downstream_df) == len(parent_df)  # Same number of rows after join
        assert ALL_SYSTEM_COLUMNS.issubset(set(downstream_df.columns))

        # Check downstream provenance structure
        provenance_schema = downstream_df.schema[METAXY_PROVENANCE_BY_FIELD]
        assert isinstance(provenance_schema, pl.Struct)
        field_names = {field.name for field in provenance_schema.fields}
        assert field_names == {"child_field"}

        # Check that provenance values are correctly truncated
        for row in downstream_df.iter_rows(named=True):
            provenance = row[METAXY_PROVENANCE_BY_FIELD]
            assert "child_field" in provenance
            assert isinstance(provenance["child_field"], str)
            assert len(provenance["child_field"]) <= 16, (
                "Hash should be truncated to 16 chars"
            )
            assert len(provenance["child_field"]) > 0, "Hash should not be empty"

        # Check version columns
        assert downstream_df[METAXY_FEATURE_VERSION].unique().to_list() == [
            feature_versions["child"]
        ]
        assert downstream_df[METAXY_SNAPSHOT_VERSION].unique().to_list() == [
            snapshot_version
        ]

    property_test()


def test_downstream_metadata_strategy_multiple_upstreams(graph: FeatureGraph) -> None:
    """Test downstream metadata with multiple upstream features."""

    class ParentA(
        Feature,
        spec=SampleFeatureSpec(
            key="parent_a",
            fields=["field_a"],
        ),
    ):
        pass

    class ParentB(
        Feature,
        spec=SampleFeatureSpec(
            key="parent_b",
            fields=["field_b"],
        ),
    ):
        pass

    class ChildFeature(
        Feature,
        spec=SampleFeatureSpec(
            key="child",
            deps=[
                FeatureDep(feature="parent_a"),
                FeatureDep(feature="parent_b"),
            ],
            fields=["result"],
        ),
    ):
        pass

    plan = graph.get_feature_plan(FeatureKey(["child"]))

    # Get versions from graph
    feature_versions = {
        "parent_a": ParentA.feature_version(),
        "parent_b": ParentB.feature_version(),
        "child": ChildFeature.feature_version(),
    }
    snapshot_version = "test_snapshot_v1"

    @given(
        downstream_metadata_strategy(
            plan,
            feature_versions=feature_versions,
            snapshot_version=snapshot_version,
            hash_algorithm=HashAlgorithm.SHA256,
            min_rows=5,
            max_rows=10,
        )
    )
    @settings(max_examples=10)
    def property_test(data: tuple[dict[str, pl.DataFrame], pl.DataFrame]) -> None:
        upstream_data, downstream_df = data

        # Check both upstream features exist
        assert set(upstream_data.keys()) == {"parent_a", "parent_b"}

        parent_a_df = upstream_data["parent_a"]
        parent_b_df = upstream_data["parent_b"]

        # All should have same row count (aligned for joins)
        assert len(parent_a_df) == len(parent_b_df) == len(downstream_df)
        assert 5 <= len(downstream_df) <= 10

        # Check downstream has correct structure
        assert ALL_SYSTEM_COLUMNS.issubset(set(downstream_df.columns))

        provenance_dtype = downstream_df.schema[METAXY_PROVENANCE_BY_FIELD]
        assert isinstance(provenance_dtype, pl.Struct)
        field_names = {field.name for field in provenance_dtype.fields}
        assert field_names == {"result"}

        # Verify provenance is calculated (non-empty hashes)
        for row in downstream_df.iter_rows(named=True):
            provenance = row[METAXY_PROVENANCE_BY_FIELD]
            assert len(provenance["result"]) > 0
            assert len(provenance["result"]) <= 32  # SHA256 truncated to 32

    property_test()


def test_downstream_metadata_strategy_no_truncation(graph: FeatureGraph) -> None:
    """Test downstream metadata without hash truncation."""

    class ParentFeature(
        Feature,
        spec=SampleFeatureSpec(
            key="parent",
            fields=["field1"],
        ),
    ):
        pass

    class ChildFeature(
        Feature,
        spec=SampleFeatureSpec(
            key="child",
            deps=[FeatureDep(feature="parent")],
            fields=["result"],
        ),
    ):
        pass

    plan = graph.get_feature_plan(FeatureKey(["child"]))

    feature_versions = {
        "parent": ParentFeature.feature_version(),
        "child": ChildFeature.feature_version(),
    }
    snapshot_version = "test_v1"

    @given(
        downstream_metadata_strategy(
            plan,
            feature_versions=feature_versions,
            snapshot_version=snapshot_version,
            hash_algorithm=HashAlgorithm.XXHASH64,
            min_rows=2,
            max_rows=5,
        )
    )
    @settings(max_examples=5)
    def property_test(data: tuple[dict[str, pl.DataFrame], pl.DataFrame]) -> None:
        _, downstream_df = data

        # Without truncation, check that hashes are non-empty strings
        for row in downstream_df.iter_rows(named=True):
            provenance = row[METAXY_PROVENANCE_BY_FIELD]
            hash_value = provenance["result"]
            # Hash should be a non-empty string
            assert isinstance(hash_value, str)
            assert len(hash_value) > 0, "Hash should not be empty"

    property_test()
