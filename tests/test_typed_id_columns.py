"""Tests for custom ID columns feature (simplified after removing type support)."""

import narwhals as nw
import polars as pl
import pytest

from metaxy.data_versioning.joiners.narwhals import NarwhalsJoiner
from metaxy.models.feature import Feature, FeatureGraph
from metaxy.models.feature_spec import FeatureDep, FeatureSpec
from metaxy.models.field import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey


def test_feature_spec_id_columns_list_only():
    """Test that FeatureSpec only accepts list of column names."""
    # List of columns (only supported format now)
    spec = FeatureSpec(
        key=FeatureKey(["test"]),
        deps=None,
        id_columns=["user_id", "timestamp"],
    )
    assert spec.id_columns == ["user_id", "timestamp"]
    assert spec.id_columns == ["user_id", "timestamp"]


def test_feature_spec_id_columns_backward_compat():
    """Test backward compatibility with list-based id_columns."""
    # List format
    spec_list = FeatureSpec(
        key=FeatureKey(["test"]),
        deps=None,
        id_columns=["user_id", "session_id"],
    )
    assert spec_list.id_columns == ["user_id", "session_id"]
    assert spec_list.id_columns == ["user_id", "session_id"]

    # None (default) - should use sample_uid
    spec_none = FeatureSpec(
        key=FeatureKey(["test"]),
        deps=None,
    )
    assert spec_none.id_columns == ["sample_uid"]


def test_feature_spec_empty_list_validation():
    """Test that empty list for id_columns raises validation error."""
    with pytest.raises(ValueError, match="id_columns must be non-empty"):
        FeatureSpec(
            key=FeatureKey(["test"]),
            deps=None,
            id_columns=[],  # Empty list should raise error
        )


def test_narwhals_joiner_empty_upstream(graph: FeatureGraph):
    """Test NarwhalsJoiner handles empty upstream refs for source features."""
    joiner = NarwhalsJoiner()

    # Create source feature with custom ID columns (just names, no types)
    class SourceFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["source"]),
            deps=None,
            id_columns=["user_uuid", "event_time", "score"],
        ),
    ):
        pass

    plan = graph.get_feature_plan(SourceFeature.spec.key)

    # No upstream refs for source feature
    joined, mapping = joiner.join_upstream(
        upstream_refs={},
        feature_spec=SourceFeature.spec,
        feature_plan=plan,
    )

    # Materialize to check schema
    result = joined.collect()

    # Check columns exist
    assert "user_uuid" in result.columns
    assert "event_time" in result.columns
    assert "score" in result.columns

    # Should be empty (source feature with no data)
    assert len(result) == 0


def test_feature_version_changes_with_different_id_columns(graph: FeatureGraph):
    """Test that feature_version changes when id_columns change."""

    # Feature with one set of ID columns
    class Feature1(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test1"]),
            deps=None,
            id_columns=["entity_id"],
        ),
    ):
        pass

    # Feature with different ID columns
    class Feature2(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test2"]),
            deps=None,
            id_columns=["user_id", "session_id"],
        ),
    ):
        pass

    # feature_spec_version should be different because id_columns differ
    assert Feature1.feature_spec_version() != Feature2.feature_spec_version()


def test_composite_key(graph: FeatureGraph):
    """Test composite key with multiple ID columns."""
    joiner = NarwhalsJoiner()

    # Create feature with composite key
    class CompositeKeyFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["composite"]),
            deps=None,
            id_columns=["tenant_id", "user_id", "timestamp", "is_active"],
        ),
    ):
        pass

    plan = graph.get_feature_plan(CompositeKeyFeature.spec.key)

    # Create empty frame
    joined, _ = joiner.join_upstream(
        upstream_refs={},
        feature_spec=CompositeKeyFeature.spec,
        feature_plan=plan,
    )

    result = joined.collect()

    # Check all columns exist
    assert "tenant_id" in result.columns
    assert "user_id" in result.columns
    assert "timestamp" in result.columns
    assert "is_active" in result.columns


def test_joining_with_custom_id_columns(graph: FeatureGraph):
    """Test that joining works correctly with custom ID columns."""
    joiner = NarwhalsJoiner()

    # Create upstream with custom ID column
    class UpstreamFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["upstream"]),
            deps=None,
            id_columns=["content_id"],
        ),
    ):
        pass

    # Create target depending on upstream with same ID column
    class TargetFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["target"]),
            deps=[FeatureDep(key=FeatureKey(["upstream"]))],
            id_columns=["content_id"],
        ),
    ):
        pass

    # Create upstream data with string IDs
    upstream_data = nw.from_native(
        pl.DataFrame(
            {
                "content_id": ["video-001", "video-002", "video-003"],
                "data_version": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                ],
                "duration": [120, 180, 90],
            }
        ).lazy()
    )

    plan = graph.get_feature_plan(TargetFeature.spec.key)

    # Join should work with custom ID column
    joined, mapping = joiner.join_upstream(
        upstream_refs={"upstream": upstream_data},
        feature_spec=TargetFeature.spec,
        feature_plan=plan,
    )

    result = joined.collect()

    # Check join succeeded
    assert len(result) == 3
    assert "content_id" in result.columns
    assert set(result["content_id"].to_list()) == {
        "video-001",
        "video-002",
        "video-003",
    }

    # Check type is preserved (String is the default for text data)
    assert str(result.schema["content_id"]) == "String"


def test_metadata_store_with_custom_id_columns(graph: FeatureGraph):
    """Test that metadata store works with custom ID columns."""
    from metaxy.metadata_store import InMemoryMetadataStore

    # Create feature with custom ID
    class CustomIDFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["custom_id"]),
            deps=None,
            id_columns=["uuid"],
            fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
        ),
    ):
        pass

    with InMemoryMetadataStore() as store:
        # Write metadata with string ID
        df = nw.from_native(
            pl.DataFrame(
                {
                    "uuid": ["uuid-123", "uuid-456", "uuid-789"],
                    "data_version": [
                        {"data": "hash1"},
                        {"data": "hash2"},
                        {"data": "hash3"},
                    ],
                    "custom_field": ["a", "b", "c"],
                }
            )
        )

        store.write_metadata(CustomIDFeature, df)

        # Read back and verify
        result = store.read_metadata(CustomIDFeature).collect()

        assert "uuid" in result.columns
        assert len(result) == 3
        assert set(result["uuid"].to_list()) == {"uuid-123", "uuid-456", "uuid-789"}
        assert (
            str(result.schema["uuid"]) == "String"
        )  # Type preserved even without explicit type spec
