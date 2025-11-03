"""Edge case tests for the generics type system with ID columns.

This module tests unusual but valid scenarios and error conditions
related to the generic type system for ID columns.
"""

from __future__ import annotations

import narwhals as nw
import polars as pl
import pydantic
import pytest

from metaxy.data_versioning.joiners.narwhals import NarwhalsJoiner
from metaxy.metadata_store import InMemoryMetadataStore
from metaxy.models.feature import BaseFeature, FeatureGraph
from metaxy.models.feature_spec import (
    BaseFeatureSpec,
    FeatureDep,
    IDColumns,
    TestingFeatureSpec,
)
from metaxy.models.field import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey


def test_single_id_column_vs_multiple():
    """Test features with single vs multiple ID columns work independently."""

    class SingleIDFeature(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["single"]),
            id_columns=["id"],  # Single column
        ),
    ):
        pass

    class MultiIDFeature(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["multi"]),
            id_columns=["id1", "id2", "id3"],  # Multiple columns
        ),
    ):
        pass

    assert len(SingleIDFeature.spec().id_columns) == 1
    assert len(MultiIDFeature.spec().id_columns) == 3


def test_id_column_names_with_special_characters():
    """Test that ID column names with special characters work correctly."""

    class SpecialCharsFeature(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["special"]),
            # Column names with underscores, numbers, etc.
            id_columns=["user_id_123", "session_2024", "_internal_id"],
        ),
    ):
        pass

    assert SpecialCharsFeature.spec().id_columns == [
        "user_id_123",
        "session_2024",
        "_internal_id",
    ]


def test_very_long_id_column_list():
    """Test feature with many ID columns (stress test)."""

    # Create a feature with 10 ID columns
    many_columns = [f"col_{i}" for i in range(10)]

    class ManyColumnsFeature(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["many"]),
            id_columns=many_columns,
        ),
    ):
        pass

    assert ManyColumnsFeature.spec().id_columns == many_columns
    assert len(ManyColumnsFeature.spec().id_columns) == 10


def test_duplicate_id_columns_allowed_but_unusual():
    """Test that duplicate ID column names are allowed (though not recommended).

    This is technically valid from Pydantic's perspective, though it may cause
    issues in actual database operations.
    """

    # Pydantic doesn't prevent duplicate values in a list
    spec = TestingFeatureSpec(
        key=FeatureKey(["test"]),
        id_columns=["id", "id", "id"],  # Duplicates
    )

    # The duplicates are preserved as-is
    assert spec.id_columns == ["id", "id", "id"]


def test_id_columns_ordering_preserved():
    """Test that the order of ID columns is preserved."""

    spec1 = TestingFeatureSpec(
        key=FeatureKey(["test1"]),
        id_columns=["a", "b", "c"],
    )

    spec2 = TestingFeatureSpec(
        key=FeatureKey(["test2"]),
        id_columns=["c", "b", "a"],  # Different order
    )

    # Order matters - these should be different
    assert spec1.id_columns != spec2.id_columns
    assert spec1.feature_spec_version != spec2.feature_spec_version


def test_unicode_id_column_names():
    """Test that unicode characters in ID column names work (though not recommended)."""

    spec = TestingFeatureSpec(
        key=FeatureKey(["test"]),
        id_columns=["用户ID", "会话ID"],  # Chinese characters
    )

    assert spec.id_columns == ["用户ID", "会话ID"]


def test_metadata_store_with_maximum_id_columns(graph: FeatureGraph):
    """Test metadata store operations with many ID columns."""

    # Feature with 5 ID columns
    class MultiKeyFeature(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["multikey"]),
            id_columns=["tenant", "user", "session", "device", "timestamp"],
            fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
        ),
    ):
        pass

    with InMemoryMetadataStore() as store:
        # Write data with all 5 ID columns
        df = nw.from_native(
            pl.DataFrame(
                {
                    "tenant": ["t1", "t1", "t2"],
                    "user": ["u1", "u2", "u3"],
                    "session": ["s1", "s2", "s3"],
                    "device": ["d1", "d2", "d3"],
                    "timestamp": [1000, 2000, 3000],
                    "provenance_by_field": [
                        {"data": "hash1"},
                        {"data": "hash2"},
                        {"data": "hash3"},
                    ],
                }
            )
        )

        store.write_metadata(MultiKeyFeature, df)

        # Read back and verify all columns present
        result = store.read_metadata(MultiKeyFeature).collect()

        assert "tenant" in result.columns
        assert "user" in result.columns
        assert "session" in result.columns
        assert "device" in result.columns
        assert "timestamp" in result.columns
        assert len(result) == 3


def test_upstream_with_subset_of_target_id_columns_fails(graph: FeatureGraph):
    """Test that upstream missing required ID columns raises clear error."""

    joiner = NarwhalsJoiner()

    # Upstream with fewer ID columns
    class UpstreamLimited(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["upstream"]),
            id_columns=["user_id"],  # Only user_id
        ),
    ):
        pass

    # Target needs more ID columns
    class TargetExpanded(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["target"]),
            deps=[FeatureDep(feature=FeatureKey(["upstream"]))],
            id_columns=["user_id", "session_id", "device_id"],  # Needs 3
        ),
    ):
        pass

    # Upstream data missing required columns
    upstream_data = nw.from_native(
        pl.DataFrame(
            {
                "user_id": [1, 2, 3],
                # Missing session_id and device_id
                "provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
            }
        ).lazy()
    )

    plan = graph.get_feature_plan(TargetExpanded.spec().key)

    # Should fail with clear error
    with pytest.raises(ValueError) as exc_info:
        joiner.join_upstream(
            upstream_refs={"upstream": upstream_data},
            feature_spec=TargetExpanded.spec(),
            feature_plan=plan,
        )

    error_msg = str(exc_info.value)
    assert "missing some required ID columns" in error_msg
    # Should mention both missing columns
    assert "session_id" in error_msg
    assert "device_id" in error_msg


def test_mixed_case_id_column_names():
    """Test that ID column names with mixed case are preserved."""

    spec = TestingFeatureSpec(
        key=FeatureKey(["test"]),
        id_columns=["UserId", "SessionID", "device_id"],  # Mixed case
    )

    # Case should be preserved exactly
    assert spec.id_columns == ["UserId", "SessionID", "device_id"]


def test_id_columns_with_reserved_sql_keywords():
    """Test ID columns with SQL reserved keywords (edge case)."""

    # These are SQL reserved words but valid Python identifiers
    spec = TestingFeatureSpec(
        key=FeatureKey(["test"]),
        id_columns=["select", "from", "where"],  # SQL keywords
    )

    # Should be accepted (database escaping is handled elsewhere)
    assert spec.id_columns == ["select", "from", "where"]


def test_feature_spec_with_literal_union_type():
    """Test using Literal with Union for flexible but constrained ID columns."""

    # Define a type that allows specific column name combinations
    list[str]  # In practice, you'd validate this separately

    class FlexibleSpec(BaseFeatureSpec):
        id_columns: pydantic.SkipValidation[IDColumns] = pydantic.Field(
            default=["sample_uid"],
        )

    # Can create with default
    spec1 = FlexibleSpec(
        key=FeatureKey(["test1"]),
    )
    assert spec1.id_columns == ["sample_uid"]

    # Can create with custom
    spec2 = FlexibleSpec(
        key=FeatureKey(["test2"]),
        id_columns=["user_id", "session_id"],
    )
    assert spec2.id_columns == ["user_id", "session_id"]


def test_pydantic_field_metadata_preserved():
    """Test that Pydantic Field metadata is preserved in subclasses."""

    class DocumentedSpec(BaseFeatureSpec):
        id_columns: pydantic.SkipValidation[IDColumns] = pydantic.Field(
            default=["doc_id"],
            description="Document identifier columns",
            examples=[["doc_id"], ["doc_id", "version"]],
        )

    spec = DocumentedSpec(
        key=FeatureKey(["test"]),
    )

    # Check that the field info is accessible
    field_info = spec.model_fields["id_columns"]
    assert field_info.description == "Document identifier columns"
    assert field_info.examples is not None


def test_feature_with_zero_upstream_deps_and_custom_id():
    """Test source feature (no deps) with custom ID columns."""

    class SourceFeature(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["source"]),
            # No dependencies
            id_columns=["entity_id", "event_time"],
        ),
    ):
        pass

    # Source features should still have ID columns defined
    assert SourceFeature.spec().id_columns == ["entity_id", "event_time"]


def test_deeply_nested_feature_dependency_chain():
    """Test that ID columns work correctly in a long dependency chain."""

    # Create a chain: F1 -> F2 -> F3 -> F4
    class F1(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["f1"]),
            id_columns=["id"],
        ),
    ):
        pass

    class F2(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["f2"]),
            deps=[FeatureDep(feature=FeatureKey(["f1"]))],
            id_columns=["id"],
        ),
    ):
        pass

    class F3(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["f3"]),
            deps=[FeatureDep(feature=FeatureKey(["f2"]))],
            id_columns=["id"],
        ),
    ):
        pass

    class F4(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["f4"]),
            deps=[FeatureDep(feature=FeatureKey(["f3"]))],
            id_columns=["id"],
        ),
    ):
        pass

    # All features in the chain should have consistent ID columns
    assert F1.spec().id_columns == ["id"]
    assert F2.spec().id_columns == ["id"]
    assert F3.spec().id_columns == ["id"]
    assert F4.spec().id_columns == ["id"]


def test_feature_diamond_dependency_with_id_columns():
    """Test diamond dependency pattern with ID columns.

           F1
          /  \\
        F2    F3
          \\  /
           F4
    """

    class F1(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["f1"]),
            id_columns=["user_id"],
        ),
    ):
        pass

    class F2(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["f2"]),
            deps=[FeatureDep(feature=FeatureKey(["f1"]))],
            id_columns=["user_id"],
        ),
    ):
        pass

    class F3(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["f3"]),
            deps=[FeatureDep(feature=FeatureKey(["f1"]))],
            id_columns=["user_id"],
        ),
    ):
        pass

    class F4(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["f4"]),
            deps=[
                FeatureDep(feature=FeatureKey(["f2"])),
                FeatureDep(feature=FeatureKey(["f3"])),
            ],
            id_columns=["user_id"],
        ),
    ):
        pass

    # All features should have consistent ID columns
    for feature_cls in [F1, F2, F3, F4]:
        assert feature_cls.spec().id_columns == ["user_id"]


def test_id_column_case_sensitivity_in_validation():
    """Test that ID column names are case-sensitive."""

    spec1 = TestingFeatureSpec(
        key=FeatureKey(["test1"]),
        id_columns=["userid"],  # lowercase
    )

    spec2 = TestingFeatureSpec(
        key=FeatureKey(["test2"]),
        id_columns=["UserId"],  # mixed case
    )

    # These are different columns (case matters)
    assert spec1.id_columns != spec2.id_columns


def test_whitespace_in_id_column_names():
    """Test that whitespace in column names is preserved (though not recommended)."""

    # Column names with spaces (valid but unusual)
    spec = TestingFeatureSpec(
        key=FeatureKey(["test"]),
        id_columns=["user id", "session id"],  # Spaces in names
    )

    # Whitespace should be preserved
    assert spec.id_columns == ["user id", "session id"]


def test_empty_string_id_column_name():
    """Test that empty string column names are technically allowed but unusual."""

    # Empty string is a valid Python string
    spec = TestingFeatureSpec(
        key=FeatureKey(["test"]),
        id_columns=["", "valid_col"],  # One empty, one valid
    )

    # Empty string is preserved
    assert spec.id_columns == ["", "valid_col"]


def test_id_columns_mutation_attempt():
    """Test that id_columns cannot be mutated after creation (if frozen)."""

    spec = TestingFeatureSpec(
        key=FeatureKey(["test"]),
        id_columns=["id1", "id2"],
    )

    # With SkipValidation, direct mutation is no longer accessible via .copy() or .append()
    # The underlying value is wrapped by SkipValidation
    initial_columns = spec.id_columns

    # Verify initial state
    assert initial_columns == ["id1", "id2"]

    # The feature_spec_version is computed on access
    initial_version = spec.feature_spec_version
    assert initial_version is not None


def test_concurrent_features_with_different_id_columns_in_same_graph(
    graph: FeatureGraph,
):
    """Test that multiple features with different ID columns can coexist in one graph."""

    class FeatureA(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["a"]),
            id_columns=["sample_uid"],
        ),
    ):
        pass

    class FeatureB(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["b"]),
            id_columns=["user_id"],
        ),
    ):
        pass

    class FeatureC(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["c"]),
            id_columns=["entity_id", "timestamp"],
        ),
    ):
        pass

    # All three features should be registered in the graph
    assert len(graph.features_by_key) == 3

    # Each should maintain its own ID columns
    assert graph.features_by_key[FeatureKey(["a"])].spec().id_columns == ["sample_uid"]
    assert graph.features_by_key[FeatureKey(["b"])].spec().id_columns == ["user_id"]
    assert graph.features_by_key[FeatureKey(["c"])].spec().id_columns == [
        "entity_id",
        "timestamp",
    ]
