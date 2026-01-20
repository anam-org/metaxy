"""Tests for custom ID columns feature (simplified after removing type support)."""

from pathlib import Path
from typing import Any

import narwhals as nw
import polars as pl
import pytest

from metaxy._testing.models import SampleFeature, SampleFeatureSpec
from metaxy._testing.pytest_helpers import add_metaxy_system_columns
from metaxy.metadata_store.delta import DeltaMetadataStore
from metaxy.models.constants import (
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.feature import FeatureGraph
from metaxy.models.feature_spec import FeatureDep
from metaxy.models.field import FieldSpec
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FeatureKey, FieldKey
from metaxy.versioning.polars import PolarsVersioningEngine


# Simple test joiner
class TestJoiner:
    """Test utility that wraps PolarsVersioningEngine."""

    def join_upstream(
        self,
        upstream_refs: dict[str, "nw.LazyFrame[Any]"],
        feature_spec: Any,
        feature_plan: FeaturePlan,
        upstream_columns: dict[str, tuple[str, ...] | None] | None = None,
        upstream_renames: dict[str, dict[str, str] | None] | None = None,
    ) -> tuple["nw.LazyFrame[Any]", dict[str, str]]:
        """Join upstream feature metadata using PolarsVersioningEngine."""
        # Handle empty upstream refs (source features)
        if not upstream_refs:
            empty_df = pl.DataFrame({col: [] for col in feature_spec.id_columns})
            empty_df = empty_df.with_columns(
                [
                    pl.lit(None)
                    .alias(METAXY_PROVENANCE_BY_FIELD)
                    .cast(pl.Struct({"default": pl.String})),
                ]
            )
            return nw.from_native(empty_df.lazy(), eager_only=False), {}

        engine = PolarsVersioningEngine(plan=feature_plan)

        upstream_by_key = {}
        for k, v in upstream_refs.items():
            df = v.collect().to_polars()
            df = add_metaxy_system_columns(df)
            upstream_by_key[FeatureKey(k)] = nw.from_native(df.lazy(), eager_only=False)

        joined = engine.prepare_upstream(upstream_by_key, filters=None)

        mapping = {}
        for upstream_key_str in upstream_refs.keys():
            upstream_key = FeatureKey(upstream_key_str)
            provenance_col_name = (
                f"{METAXY_PROVENANCE_BY_FIELD}{upstream_key.to_column_suffix()}"
            )
            mapping[upstream_key_str] = provenance_col_name

        return joined, mapping  # ty: ignore[invalid-return-type]


def test_feature_spec_id_columns_list_only():
    """Test that SampleFeatureSpec only accepts list of column names."""
    # List of columns (only supported format now)
    spec = SampleFeatureSpec(
        key=FeatureKey(["test"]),
        id_columns=["user_id", "timestamp"],
    )
    assert spec.id_columns == ["user_id", "timestamp"]
    assert spec.id_columns == ["user_id", "timestamp"]


def test_feature_spec_id_columns_backward_compat():
    """Test backward compatibility with list-based id_columns."""
    # List format
    spec_list = SampleFeatureSpec(
        key=FeatureKey(["test"]),
        id_columns=["user_id", "session_id"],
    )
    assert spec_list.id_columns == ["user_id", "session_id"]
    assert spec_list.id_columns == ["user_id", "session_id"]

    # None (default) - should use sample_uid
    spec_none = SampleFeatureSpec(
        key=FeatureKey(["test"]),
    )
    assert spec_none.id_columns == ["sample_uid"]


def test_feature_spec_empty_list_validation():
    """Test that empty list for id_columns raises validation error."""
    with pytest.raises(ValueError, match="id_columns must be non-empty"):
        SampleFeatureSpec(
            key=FeatureKey(["test"]),
            id_columns=[],  # Empty list should raise error
        )


def test_narwhals_joiner_empty_upstream(graph: FeatureGraph):
    """Test NarwhalsJoiner handles empty upstream refs for source features."""
    joiner = TestJoiner()

    # Create source feature with custom ID columns (just names, no types)
    class SourceFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["source"]),
            id_columns=["user_uuid", "event_time", "score"],
        ),
    ):
        pass

    plan = graph.get_feature_plan(SourceFeature.spec().key)

    # No upstream refs for source feature
    joined, mapping = joiner.join_upstream(
        upstream_refs={},
        feature_spec=SourceFeature.spec(),
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
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test1"]),
            id_columns=["entity_id"],
        ),
    ):
        pass

    # Feature with different ID columns
    class Feature2(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test2"]),
            id_columns=["user_id", "session_id"],
        ),
    ):
        pass

    # feature_spec_version should be different because id_columns differ
    assert Feature1.feature_spec_version() != Feature2.feature_spec_version()


def test_composite_key(graph: FeatureGraph):
    """Test composite key with multiple ID columns."""
    joiner = TestJoiner()

    # Create feature with composite key
    class CompositeKeyFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["composite"]),
            id_columns=["tenant_id", "user_id", "timestamp", "is_active"],
        ),
    ):
        pass

    plan = graph.get_feature_plan(CompositeKeyFeature.spec().key)

    # Create empty frame
    joined, _ = joiner.join_upstream(
        upstream_refs={},
        feature_spec=CompositeKeyFeature.spec(),
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
    joiner = TestJoiner()

    # Create upstream with custom ID column
    class UpstreamFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream"]),
            id_columns=["content_id"],
        ),
    ):
        pass

    # Create target depending on upstream with same ID column
    class TargetFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["target"]),
            deps=[FeatureDep(feature=FeatureKey(["upstream"]))],
            id_columns=["content_id"],
        ),
    ):
        pass

    # Create upstream data with string IDs
    upstream_data = nw.from_native(
        pl.DataFrame(
            {
                "content_id": ["video-001", "video-002", "video-003"],
                "metaxy_provenance_by_field": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                ],
                "duration": [120, 180, 90],
            }
        ).lazy()
    )

    plan = graph.get_feature_plan(TargetFeature.spec().key)

    # Join should work with custom ID column
    joined, mapping = joiner.join_upstream(
        upstream_refs={"upstream": upstream_data},
        feature_spec=TargetFeature.spec(),
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


def test_metadata_store_with_custom_id_columns(graph: FeatureGraph, tmp_path: Path):
    """Test that metadata store works with custom ID columns."""

    # Create feature with custom ID
    class CustomIDFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["custom_id"]),
            id_columns=["uuid"],
            fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
        ),
    ):
        pass

    with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
        # Write metadata with string ID
        df = nw.from_native(
            pl.DataFrame(
                {
                    "uuid": ["uuid-123", "uuid-456", "uuid-789"],
                    "metaxy_provenance_by_field": [
                        {"data": "hash1"},
                        {"data": "hash2"},
                        {"data": "hash3"},
                    ],
                    "metaxy_data_version_by_field": [
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
