"""Tests for configurable ID columns feature."""

from pathlib import Path
from typing import Any

import narwhals as nw
import polars as pl
import pytest
from metaxy_testing.models import SampleFeature, SampleFeatureSpec
from metaxy_testing.pytest_helpers import add_metaxy_system_columns

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


# Simple test joiner that uses VersioningEngine
class TestJoiner:
    """Test utility that wraps PolarsVersioningEngine for ID columns tests."""

    def join_upstream(
        self,
        upstream_refs: dict[str, "nw.LazyFrame[Any]"],
        feature_spec: Any,
        feature_plan: FeaturePlan,
        upstream_columns: dict[str, tuple[str, ...] | None] | None = None,
        upstream_renames: dict[str, dict[str, str] | None] | None = None,
    ) -> tuple["nw.LazyFrame[Any]", dict[str, str]]:
        """Join upstream feature metadata using PolarsVersioningEngine.

        Args:
            upstream_refs: Mapping of upstream feature key strings to lazy frames
            feature_spec: Feature specification
            feature_plan: Feature plan
            upstream_columns: Optional column selection per upstream (for compatibility)
            upstream_renames: Optional column renames per upstream (for compatibility)

        Returns:
            Tuple of (joined data, mapping of upstream keys to provenance columns)
        """
        # Handle empty upstream refs (source features)
        if not upstream_refs:
            # Return empty dataframe with ID columns from feature_spec
            empty_df = pl.DataFrame({col: [] for col in feature_spec.id_columns})
            # Add required system columns
            empty_df = empty_df.with_columns(
                [
                    pl.lit(None).alias(METAXY_PROVENANCE_BY_FIELD).cast(pl.Struct({"default": pl.String})),
                ]
            )
            return nw.from_native(empty_df.lazy(), eager_only=False), {}

        # Create a PolarsVersioningEngine for this feature
        engine = PolarsVersioningEngine(plan=feature_plan)

        # Convert string keys back to FeatureKey objects and ensure data is materialized
        upstream_by_key = {}
        for k, v in upstream_refs.items():
            # Materialize the lazy frame and ensure it has all metaxy system columns
            df = v.collect().to_polars()
            df = add_metaxy_system_columns(df)
            upstream_by_key[FeatureKey(k)] = nw.from_native(df.lazy(), eager_only=False)

        # Prepare upstream (handles filtering, selecting, renaming, and joining)
        joined = engine.prepare_upstream(upstream_by_key, filters=None)

        # Build the mapping of upstream_key -> provenance_by_field column name
        # The new naming convention is: {column_name}{feature_key.to_column_suffix()}
        mapping = {}
        for upstream_key_str in upstream_refs.keys():
            upstream_key = FeatureKey(upstream_key_str)
            # The provenance_by_field column is renamed using to_column_suffix()
            provenance_col_name = f"{METAXY_PROVENANCE_BY_FIELD}{upstream_key.to_column_suffix()}"
            mapping[upstream_key_str] = provenance_col_name

        return joined, mapping  # ty: ignore[invalid-return-type]


def test_feature_spec_id_columns_default():
    """Test that id_columns defaults to None and is interpreted as ["sample_uid"]."""
    spec = SampleFeatureSpec(
        key=FeatureKey(["test"]),
    )
    assert spec.id_columns == ["sample_uid"]


def test_feature_spec_id_columns_custom():
    """Test that custom id_columns can be specified."""
    spec = SampleFeatureSpec(
        key=FeatureKey(["test"]),
        id_columns=["user_id", "session_id"],
    )
    assert spec.id_columns == ["user_id", "session_id"]


def test_feature_spec_id_columns_validation():
    """Test that empty id_columns raises validation error."""
    with pytest.raises(ValueError, match="id_columns must be non-empty"):
        SampleFeatureSpec(
            key=FeatureKey(["test"]),
            id_columns=[],  # Empty list should raise error
        )


def test_narwhals_joiner_default_id_columns(graph: FeatureGraph):
    """Test NarwhalsJoiner uses default sample_uid when id_columns not specified."""
    joiner = TestJoiner()

    # Create feature with default ID columns
    class UpstreamFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream"]),
        ),
    ):
        pass

    class TargetFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["target"]),
            deps=[FeatureDep(feature=FeatureKey(["upstream"]))],
        ),
    ):
        pass

    # Create upstream metadata with sample_uid
    upstream_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
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
                "extra_column": ["a", "b", "c"],
            }
        ).lazy()
    )

    plan = graph.get_feature_plan(TargetFeature.spec().key)

    joined, mapping = joiner.join_upstream(
        upstream_refs={"upstream": upstream_metadata},
        feature_spec=TargetFeature.spec(),
        feature_plan=plan,
    )

    result = joined.collect()

    # Should have sample_uid and all columns
    assert "sample_uid" in result.columns
    assert len(result) == 3
    assert set(result["sample_uid"].to_list()) == {1, 2, 3}


def test_narwhals_joiner_custom_single_id_column(graph: FeatureGraph):
    """Test NarwhalsJoiner with a single custom ID column."""
    joiner = TestJoiner()

    # Create features with custom ID column
    class UpstreamFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream"]),
            id_columns=["user_id"],
        ),
    ):
        pass

    class TargetFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["target"]),
            deps=[FeatureDep(feature=FeatureKey(["upstream"]))],
            id_columns=["user_id"],
        ),
    ):
        pass

    # Create upstream metadata with user_id
    upstream_metadata = nw.from_native(
        pl.DataFrame(
            {
                "user_id": [100, 200, 300],
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
                "user_name": ["Alice", "Bob", "Charlie"],
            }
        ).lazy()
    )

    plan = graph.get_feature_plan(TargetFeature.spec().key)

    joined, mapping = joiner.join_upstream(
        upstream_refs={"upstream": upstream_metadata},
        feature_spec=TargetFeature.spec(),
        feature_plan=plan,
    )

    result = joined.collect()

    # Should have user_id instead of sample_uid
    assert "user_id" in result.columns
    assert "sample_uid" not in result.columns
    assert len(result) == 3
    assert set(result["user_id"].to_list()) == {100, 200, 300}


def test_narwhals_joiner_composite_key(graph: FeatureGraph):
    """Test NarwhalsJoiner with composite key (multiple ID columns)."""
    joiner = TestJoiner()

    # Create features with composite key
    class Upstream1(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream1"]),
            id_columns=["user_id", "session_id"],
        ),
    ):
        pass

    class Upstream2(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream2"]),
            id_columns=["user_id", "session_id"],
        ),
    ):
        pass

    class TargetFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["target"]),
            deps=[
                FeatureDep(feature=FeatureKey(["upstream1"])),
                FeatureDep(feature=FeatureKey(["upstream2"])),
            ],
            id_columns=["user_id", "session_id"],
        ),
    ):
        pass

    # Create upstream metadata with composite keys
    upstream1_metadata = nw.from_native(
        pl.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "session_id": [10, 20, 10, 30],
                "metaxy_provenance_by_field": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                    {"default": "hash4"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                    {"default": "hash4"},
                ],
                "data1": ["a", "b", "c", "d"],
            }
        ).lazy()
    )

    upstream2_metadata = nw.from_native(
        pl.DataFrame(
            {
                "user_id": [1, 1, 2, 3],
                "session_id": [10, 20, 10, 40],
                "metaxy_provenance_by_field": [
                    {"default": "hash5"},
                    {"default": "hash6"},
                    {"default": "hash7"},
                    {"default": "hash8"},
                ],
                "metaxy_data_version_by_field": [
                    {"default": "hash5"},
                    {"default": "hash6"},
                    {"default": "hash7"},
                    {"default": "hash8"},
                ],
                "data2": ["w", "x", "y", "z"],
            }
        ).lazy()
    )

    plan = graph.get_feature_plan(TargetFeature.spec().key)

    joined, mapping = joiner.join_upstream(
        upstream_refs={
            "upstream1": upstream1_metadata,
            "upstream2": upstream2_metadata,
        },
        feature_spec=TargetFeature.spec(),
        feature_plan=plan,
    )

    result = joined.collect().sort(["user_id", "session_id"])

    # Should join on both user_id and session_id
    assert "user_id" in result.columns
    assert "session_id" in result.columns
    assert "sample_uid" not in result.columns

    # Inner join: only rows with matching composite keys in both upstreams
    # Matching rows: (1,10), (1,20), (2,10)
    assert len(result) == 3

    expected_rows = [(1, 10), (1, 20), (2, 10)]
    actual_rows = list(zip(result["user_id"].to_list(), result["session_id"].to_list()))
    assert actual_rows == expected_rows


def test_narwhals_joiner_empty_upstream_custom_id(graph: FeatureGraph):
    """Test NarwhalsJoiner with no upstream deps and custom ID columns."""
    joiner = TestJoiner()

    # Create source feature with custom ID columns
    class SourceFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["source"]),
            id_columns=["entity_id", "timestamp"],
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

    result = joined.collect()

    # Should have empty DataFrame with the custom ID columns
    assert "entity_id" in result.columns
    assert "timestamp" in result.columns
    assert "sample_uid" not in result.columns
    assert len(result) == 0


def test_feature_spec_version_includes_id_columns():
    """Test that feature_spec_version changes when id_columns change."""

    # Create two specs with same everything except id_columns
    spec1 = SampleFeatureSpec(
        key=FeatureKey(["test"]),
        # Uses default id_columns (None)
    )

    spec2 = SampleFeatureSpec(
        key=FeatureKey(["test"]),
        id_columns=["user_id"],  # Custom id_columns
    )

    # The feature_spec_version should be different because id_columns is different
    # (feature_spec_version includes ALL properties of the spec)
    assert spec1.feature_spec_version != spec2.feature_spec_version

    # But if we create another spec with same id_columns, versions should match
    spec3 = SampleFeatureSpec(
        key=FeatureKey(["test"]),  # Same key
        id_columns=["user_id"],  # Same id_columns as spec2
    )

    assert spec2.feature_spec_version == spec3.feature_spec_version


def test_metadata_store_integration_with_custom_id_columns(graph: FeatureGraph, tmp_path: Path):
    """Test full metadata store integration with custom ID columns."""

    # Create features with custom ID columns
    class UserFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["user"]),
            id_columns=["user_id"],
            fields=[
                FieldSpec(key=FieldKey(["profile"]), code_version="1"),
            ],
        ),
    ):
        pass

    class SessionFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["session"]),
            deps=[FeatureDep(feature=FeatureKey(["user"]))],
            id_columns=["user_id", "session_id"],
            fields=[
                FieldSpec(key=FieldKey(["activity"]), code_version="1"),
            ],
        ),
    ):
        pass

    with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
        # Write user feature metadata with user_id
        user_df = nw.from_native(
            pl.DataFrame(
                {
                    "user_id": [100, 200, 300],
                    "metaxy_provenance_by_field": [
                        {"profile": "user_hash1"},
                        {"profile": "user_hash2"},
                        {"profile": "user_hash3"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"profile": "user_hash1"},
                        {"profile": "user_hash2"},
                        {"profile": "user_hash3"},
                    ],
                    "username": ["alice", "bob", "charlie"],
                }
            )
        )
        store.write_metadata(UserFeature, user_df)

        # Read it back and verify
        read_user = store.read_metadata(UserFeature).collect()
        assert "user_id" in read_user.columns
        assert "sample_uid" not in read_user.columns
        assert len(read_user) == 3

        # Now write session feature with composite key
        session_df = nw.from_native(
            pl.DataFrame(
                {
                    "user_id": [100, 100, 200],
                    "session_id": [1, 2, 1],
                    "metaxy_provenance_by_field": [
                        {"activity": "session_hash1"},
                        {"activity": "session_hash2"},
                        {"activity": "session_hash3"},
                    ],
                    "metaxy_data_version_by_field": [
                        {"activity": "session_hash1"},
                        {"activity": "session_hash2"},
                        {"activity": "session_hash3"},
                    ],
                    "duration": [120, 180, 90],
                }
            )
        )
        store.write_metadata(SessionFeature, session_df)

        # Read session metadata
        read_session = store.read_metadata(SessionFeature).collect()
        assert "user_id" in read_session.columns
        assert "session_id" in read_session.columns
        assert "sample_uid" not in read_session.columns
        assert len(read_session) == 3


def test_feature_version_stability_with_id_columns(graph: FeatureGraph):
    """Test that feature_version is different when id_columns change.

    Since id_columns affect how features join their upstream dependencies,
    changing them changes the feature_version, which triggers migrations.
    This is the correct behavior - id_columns is part of the computational spec.
    """

    # Create feature with default ID columns
    class Feature1(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test1"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            # id_columns=None (default)
        ),
    ):
        pass

    # Create identical feature but with explicit custom ID columns
    class Feature2(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test2"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            id_columns=["user_id"],  # Custom ID columns
        ),
    ):
        pass

    # feature_version should be DIFFERENT because id_columns affects
    # the computational behavior (how rows are joined)
    version1 = Feature1.feature_version()
    version2 = Feature2.feature_version()

    # NOTE: These are DIFFERENT because id_columns is part of feature_spec_version,
    # which is hashed into feature_version computation
    assert version1 != version2

    # And feature_spec_version should also differ
    spec_version1 = Feature1.feature_spec_version()
    spec_version2 = Feature2.feature_spec_version()
    assert spec_version1 != spec_version2


def test_snapshot_stability_with_id_columns(snapshot):
    """Snapshot test to ensure id_columns in feature_spec_version is stable."""

    # Create specs with different id_columns configurations
    specs = {
        "default": SampleFeatureSpec(
            key=FeatureKey(["test"]),
            # id_columns=None (default)
        ),
        "single_custom": SampleFeatureSpec(
            key=FeatureKey(["test"]),
            id_columns=["user_id"],
        ),
        "composite": SampleFeatureSpec(
            key=FeatureKey(["test"]),
            id_columns=["user_id", "session_id"],
        ),
    }

    # Snapshot the feature_spec_versions
    spec_versions = {name: spec.feature_spec_version for name, spec in specs.items()}

    assert spec_versions == snapshot


def test_joiner_preserves_all_id_columns_in_result(graph: FeatureGraph):
    """Test that joiner result includes all ID columns from the feature spec."""
    joiner = TestJoiner()

    class TripleKeyFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["triple"]),
            id_columns=["tenant_id", "user_id", "event_id"],
        ),
    ):
        pass

    # Create empty upstream (source feature)
    plan = graph.get_feature_plan(TripleKeyFeature.spec().key)

    joined, mapping = joiner.join_upstream(
        upstream_refs={},
        feature_spec=TripleKeyFeature.spec(),
        feature_plan=plan,
    )

    result = joined.collect()

    # All three ID columns should be present
    assert "tenant_id" in result.columns
    assert "user_id" in result.columns
    assert "event_id" in result.columns
    assert "sample_uid" not in result.columns

    # Should be empty (source feature with no data)
    assert len(result) == 0


def test_backwards_compatibility_default_id_columns(graph: FeatureGraph, tmp_path: Path):
    """Test that features without explicit id_columns still use sample_uid."""

    # Create feature WITHOUT specifying id_columns (backwards compatibility)
    class LegacyFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["legacy"]),
            fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            # No id_columns specified - should default to ["sample_uid"]
        ),
    ):
        pass

    # Verify id_columns returns default
    assert LegacyFeature.spec().id_columns == ["sample_uid"]

    # Test with metadata store
    with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
        df = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
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
                    "value": [10, 20, 30],
                }
            )
        )
        store.write_metadata(LegacyFeature, df)

        result = store.read_metadata(LegacyFeature).collect()
        assert "sample_uid" in result.columns
        assert len(result) == 3
