"""Tests for configurable ID columns feature."""

import narwhals as nw
import polars as pl
import pytest

from metaxy._testing import DEFAULT_ID_COLUMNS
from metaxy.data_versioning.calculators.polars import PolarsDataVersionCalculator
from metaxy.data_versioning.diff.narwhals import NarwhalsDiffResolver
from metaxy.data_versioning.joiners.narwhals import NarwhalsJoiner
from metaxy.models.feature import Feature, FeatureGraph
from metaxy.models.feature_spec import FeatureDep, FeatureSpec
from metaxy.models.field import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey


def test_feature_spec_id_columns_default():
    """Test that id_columns defaults to None and is interpreted as ["sample_uid"]."""
    spec = FeatureSpec(
        key=FeatureKey(["test"]),
        deps=None,
    )
    assert spec.id_columns == ["sample_uid"]


def test_feature_spec_id_columns_custom():
    """Test that custom id_columns can be specified."""
    spec = FeatureSpec(
        key=FeatureKey(["test"]),
        deps=None,
        id_columns=["user_id", "session_id"],
    )
    assert spec.id_columns == ["user_id", "session_id"]


def test_feature_spec_id_columns_validation():
    """Test that empty id_columns raises validation error."""
    with pytest.raises(ValueError, match="id_columns must be non-empty"):
        FeatureSpec(
            key=FeatureKey(["test"]),
            deps=None,
            id_columns=[],  # Empty list should raise error
        )


def test_feature_id_columns_classmethod(graph: FeatureGraph):
    """Test Feature.id_columns() classmethod returns correct values."""

    # Default case
    class DefaultFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["default"]),
            deps=None,
        ),
    ):
        pass

    assert DefaultFeature.id_columns() == DEFAULT_ID_COLUMNS

    # Custom ID columns
    class CustomFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["custom"]),
            deps=None,
            id_columns=["user_id", "session_id"],
        ),
    ):
        pass

    assert CustomFeature.id_columns() == ["user_id", "session_id"]


def test_narwhals_joiner_default_id_columns(graph: FeatureGraph):
    """Test NarwhalsJoiner uses default sample_uid when id_columns not specified."""
    joiner = NarwhalsJoiner()

    # Create feature with default ID columns
    class UpstreamFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["upstream"]),
            deps=None,
        ),
    ):
        pass

    class TargetFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["target"]),
            deps=[FeatureDep(key=FeatureKey(["upstream"]))],
        ),
    ):
        pass

    # Create upstream metadata with sample_uid
    upstream_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "data_version": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                ],
                "extra_column": ["a", "b", "c"],
            }
        ).lazy()
    )

    plan = graph.get_feature_plan(TargetFeature.spec.key)

    joined, mapping = joiner.join_upstream(
        upstream_refs={"upstream": upstream_metadata},
        feature_spec=TargetFeature.spec,
        feature_plan=plan,
    )

    result = joined.collect()

    # Should have sample_uid and all columns
    assert "sample_uid" in result.columns
    assert len(result) == 3
    assert set(result["sample_uid"].to_list()) == {1, 2, 3}


def test_narwhals_joiner_custom_single_id_column(graph: FeatureGraph):
    """Test NarwhalsJoiner with a single custom ID column."""
    joiner = NarwhalsJoiner()

    # Create features with custom ID column
    class UpstreamFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["upstream"]),
            deps=None,
            id_columns=["user_id"],
        ),
    ):
        pass

    class TargetFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["target"]),
            deps=[FeatureDep(key=FeatureKey(["upstream"]))],
            id_columns=["user_id"],
        ),
    ):
        pass

    # Create upstream metadata with user_id
    upstream_metadata = nw.from_native(
        pl.DataFrame(
            {
                "user_id": [100, 200, 300],
                "data_version": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                ],
                "user_name": ["Alice", "Bob", "Charlie"],
            }
        ).lazy()
    )

    plan = graph.get_feature_plan(TargetFeature.spec.key)

    joined, mapping = joiner.join_upstream(
        upstream_refs={"upstream": upstream_metadata},
        feature_spec=TargetFeature.spec,
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
    joiner = NarwhalsJoiner()

    # Create features with composite key
    class Upstream1(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["upstream1"]),
            deps=None,
            id_columns=["user_id", "session_id"],
        ),
    ):
        pass

    class Upstream2(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["upstream2"]),
            deps=None,
            id_columns=["user_id", "session_id"],
        ),
    ):
        pass

    class TargetFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["target"]),
            deps=[
                FeatureDep(key=FeatureKey(["upstream1"])),
                FeatureDep(key=FeatureKey(["upstream2"])),
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
                "data_version": [
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
                "data_version": [
                    {"default": "hash5"},
                    {"default": "hash6"},
                    {"default": "hash7"},
                    {"default": "hash8"},
                ],
                "data2": ["w", "x", "y", "z"],
            }
        ).lazy()
    )

    plan = graph.get_feature_plan(TargetFeature.spec.key)

    joined, mapping = joiner.join_upstream(
        upstream_refs={
            "upstream1": upstream1_metadata,
            "upstream2": upstream2_metadata,
        },
        feature_spec=TargetFeature.spec,
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
    joiner = NarwhalsJoiner()

    # Create source feature with custom ID columns
    class SourceFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["source"]),
            deps=None,
            id_columns=["entity_id", "timestamp"],
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

    result = joined.collect()

    # Should have empty DataFrame with the custom ID columns
    assert "entity_id" in result.columns
    assert "timestamp" in result.columns
    assert "sample_uid" not in result.columns
    assert len(result) == 0


def test_full_pipeline_custom_id_columns(graph: FeatureGraph):
    """Test full pipeline (join -> calculate -> diff) with custom ID columns."""

    # Create features with custom ID columns
    class VideoFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["video"]),
            deps=None,
            id_columns=["content_id"],
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version=1),
            ],
        ),
    ):
        pass

    class ProcessedFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["processed"]),
            deps=[FeatureDep(key=FeatureKey(["video"]))],
            id_columns=["content_id"],
            fields=[
                FieldSpec(key=FieldKey(["analysis"]), code_version=1),
            ],
        ),
    ):
        pass

    # Step 1: Join upstream
    joiner = NarwhalsJoiner()

    video_metadata = nw.from_native(
        pl.DataFrame(
            {
                "content_id": ["vid_001", "vid_002", "vid_003"],
                "data_version": [
                    {"frames": "frame_v1"},
                    {"frames": "frame_v2"},
                    {"frames": "frame_v3"},
                ],
                "duration": [120, 180, 90],
            }
        ).lazy()
    )

    plan = graph.get_feature_plan(ProcessedFeature.spec.key)

    joined, mapping = joiner.join_upstream(
        upstream_refs={"video": video_metadata},
        feature_spec=ProcessedFeature.spec,
        feature_plan=plan,
    )

    # Step 2: Calculate data versions
    calculator = PolarsDataVersionCalculator()

    with_versions = calculator.calculate_data_versions(
        joined_upstream=joined,
        feature_spec=ProcessedFeature.spec,
        feature_plan=plan,
        upstream_column_mapping=mapping,
    )

    # Step 3: Diff with current
    diff_resolver = NarwhalsDiffResolver()

    current = nw.from_native(
        pl.DataFrame(
            {
                "content_id": ["vid_001", "vid_002"],
                "data_version": [
                    {"analysis": "old_hash1"},
                    {"analysis": "old_hash2"},
                ],
            }
        ).lazy()
    )

    diff_result = diff_resolver.find_changes(
        target_versions=with_versions,
        current_metadata=current,
        id_columns=ProcessedFeature.id_columns(),  # Pass ID columns explicitly
    )

    # Check results
    added = diff_result.added.collect()
    changed = diff_result.changed.collect()
    removed = diff_result.removed.collect()

    # Added: vid_003 (not in current)
    assert len(added) == 1
    assert added["content_id"][0] == "vid_003"

    # Changed: vid_001 and vid_002 (different hashes)
    assert len(changed) == 2
    assert set(changed["content_id"].to_list()) == {"vid_001", "vid_002"}

    # Removed: none
    assert len(removed) == 0


def test_feature_spec_version_includes_id_columns():
    """Test that feature_spec_version changes when id_columns change."""

    # Create two specs with same everything except id_columns
    spec1 = FeatureSpec(
        key=FeatureKey(["test"]),
        deps=None,
        # Uses default id_columns (None)
    )

    spec2 = FeatureSpec(
        key=FeatureKey(["test"]),
        deps=None,
        id_columns=["user_id"],  # Custom id_columns
    )

    # The feature_spec_version should be different because id_columns is different
    # (feature_spec_version includes ALL properties of the spec)
    assert spec1.feature_spec_version != spec2.feature_spec_version

    # But if we create another spec with same id_columns, versions should match
    spec3 = FeatureSpec(
        key=FeatureKey(["test"]),  # Same key
        deps=None,
        id_columns=["user_id"],  # Same id_columns as spec2
    )

    assert spec2.feature_spec_version == spec3.feature_spec_version


def test_mixed_id_columns_behavior(graph: FeatureGraph):
    """Test validation when upstream and target features have different ID columns.

    The joiner now validates that all upstream features have the required ID columns
    from the target feature and raises a clear error if any are missing.
    """
    joiner = NarwhalsJoiner()

    # Create upstream with one set of ID columns
    class UpstreamFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["upstream"]),
            deps=None,
            id_columns=["user_id"],  # Upstream declares user_id
        ),
    ):
        pass

    # Create target with different ID columns
    class TargetFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["target"]),
            deps=[FeatureDep(key=FeatureKey(["upstream"]))],
            id_columns=["user_id", "session_id"],  # Target needs both columns
        ),
    ):
        pass

    # Scenario 1: Upstream has all required columns - should work
    upstream_with_both = nw.from_native(
        pl.DataFrame(
            {
                "user_id": [1, 2, 3],
                "session_id": [10, 20, 30],  # Has both columns
                "data_version": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                ],
            }
        ).lazy()
    )

    plan = graph.get_feature_plan(TargetFeature.spec.key)

    # This should work because upstream has both columns
    joined, mapping = joiner.join_upstream(
        upstream_refs={"upstream": upstream_with_both},
        feature_spec=TargetFeature.spec,
        feature_plan=plan,
    )
    result = joined.collect()
    assert len(result) == 3  # All rows joined successfully

    # Scenario 2: Upstream missing required ID columns - should raise clear error
    upstream_missing_column = nw.from_native(
        pl.DataFrame(
            {
                "user_id": [1, 2, 3],
                # Missing session_id!
                "data_version": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                ],
            }
        ).lazy()
    )

    # Should raise clear validation error about missing session_id
    with pytest.raises(ValueError) as exc_info:
        joined2, _ = joiner.join_upstream(
            upstream_refs={"upstream": upstream_missing_column},
            feature_spec=TargetFeature.spec,
            feature_plan=plan,
        )

    # Check the error message is helpful
    error_msg = str(exc_info.value)
    assert "missing required ID columns" in error_msg
    assert "session_id" in error_msg
    assert "upstream" in error_msg

    # Scenario 3: With multiple upstreams, all must have required ID columns
    class Upstream2(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["upstream2"]),
            deps=None,
            id_columns=["user_id"],
        ),
    ):
        pass

    class MultiUpstreamTarget(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["multi"]),
            deps=[
                FeatureDep(key=FeatureKey(["upstream"])),
                FeatureDep(key=FeatureKey(["upstream2"])),
            ],
            id_columns=["user_id", "session_id"],  # Needs both
        ),
    ):
        pass

    # Both upstreams have user_id but only one has session_id
    upstream2_data = nw.from_native(
        pl.DataFrame(
            {
                "user_id": [1, 2, 3],
                "session_id": [10, 20, 30],  # This one has session_id
                "data_version": [
                    {"default": "hash4"},
                    {"default": "hash5"},
                    {"default": "hash6"},
                ],
            }
        ).lazy()
    )

    plan2 = graph.get_feature_plan(MultiUpstreamTarget.spec.key)

    # This will fail because upstream (first) doesn't have session_id
    with pytest.raises(ValueError) as exc_info:
        joined3, _ = joiner.join_upstream(
            upstream_refs={
                "upstream": upstream_missing_column,  # Missing session_id
                "upstream2": upstream2_data,  # Has session_id
            },
            feature_spec=MultiUpstreamTarget.spec,
            feature_plan=plan2,
        )

    # Check error mentions the specific upstream missing the column
    error_msg = str(exc_info.value)
    assert "upstream" in error_msg
    assert "missing required ID columns" in error_msg
    assert "session_id" in error_msg


def test_metadata_store_integration_with_custom_id_columns(graph: FeatureGraph):
    """Test full metadata store integration with custom ID columns."""
    from metaxy.metadata_store import InMemoryMetadataStore

    # Create features with custom ID columns
    class UserFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["user"]),
            deps=None,
            id_columns=["user_id"],
            fields=[
                FieldSpec(key=FieldKey(["profile"]), code_version=1),
            ],
        ),
    ):
        pass

    class SessionFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["session"]),
            deps=[FeatureDep(key=FeatureKey(["user"]))],
            id_columns=["user_id", "session_id"],
            fields=[
                FieldSpec(key=FieldKey(["activity"]), code_version=1),
            ],
        ),
    ):
        pass

    with InMemoryMetadataStore() as store:
        # Write user feature metadata with user_id
        user_df = nw.from_native(
            pl.DataFrame(
                {
                    "user_id": [100, 200, 300],
                    "data_version": [
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
                    "data_version": [
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
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test1"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            # id_columns=None (default)
        ),
    ):
        pass

    # Create identical feature but with explicit custom ID columns
    class Feature2(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test2"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
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


def test_diff_resolver_with_composite_id_columns():
    """Test NarwhalsDiffResolver with composite ID columns."""
    diff_resolver = NarwhalsDiffResolver()

    # Target with composite key
    target = nw.from_native(
        pl.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "session_id": [10, 20, 10, 30],
                "data_version": [
                    {"default": "new1"},
                    {"default": "new2"},
                    {"default": "new3"},
                    {"default": "new4"},
                ],
            }
        ).lazy()
    )

    # Current with some overlapping composite keys
    current = nw.from_native(
        pl.DataFrame(
            {
                "user_id": [1, 1, 2],
                "session_id": [10, 20, 20],  # Note: (2, 20) not in target
                "data_version": [
                    {"default": "old1"},  # Changed
                    {"default": "new2"},  # Unchanged
                    {"default": "old3"},  # Will be removed
                ],
            }
        ).lazy()
    )

    result = diff_resolver.find_changes(
        target_versions=target,
        current_metadata=current,
        id_columns=["user_id", "session_id"],  # Composite key
    )

    # Materialize results
    added = result.added.collect().sort(["user_id", "session_id"])
    changed = result.changed.collect().sort(["user_id", "session_id"])
    removed = result.removed.collect().sort(["user_id", "session_id"])

    # Added: (2, 10) and (2, 30) - not in current
    assert len(added) == 2
    added_keys = list(zip(added["user_id"].to_list(), added["session_id"].to_list()))
    assert added_keys == [(2, 10), (2, 30)]

    # Changed: (1, 10) - different hash
    assert len(changed) == 1
    assert changed["user_id"][0] == 1
    assert changed["session_id"][0] == 10

    # Removed: (2, 20) - not in target
    assert len(removed) == 1
    assert removed["user_id"][0] == 2
    assert removed["session_id"][0] == 20


def test_snapshot_stability_with_id_columns(snapshot):
    """Snapshot test to ensure id_columns in feature_spec_version is stable."""

    # Create specs with different id_columns configurations
    specs = {
        "default": FeatureSpec(
            key=FeatureKey(["test"]),
            deps=None,
            # id_columns=None (default)
        ),
        "single_custom": FeatureSpec(
            key=FeatureKey(["test"]),
            deps=None,
            id_columns=["user_id"],
        ),
        "composite": FeatureSpec(
            key=FeatureKey(["test"]),
            deps=None,
            id_columns=["user_id", "session_id"],
        ),
    }

    # Snapshot the feature_spec_versions
    spec_versions = {name: spec.feature_spec_version for name, spec in specs.items()}

    assert spec_versions == snapshot


def test_joiner_preserves_all_id_columns_in_result(graph: FeatureGraph):
    """Test that joiner result includes all ID columns from the feature spec."""
    joiner = NarwhalsJoiner()

    class TripleKeyFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["triple"]),
            deps=None,
            id_columns=["tenant_id", "user_id", "event_id"],
        ),
    ):
        pass

    # Create empty upstream (source feature)
    plan = graph.get_feature_plan(TripleKeyFeature.spec.key)

    joined, mapping = joiner.join_upstream(
        upstream_refs={},
        feature_spec=TripleKeyFeature.spec,
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


def test_id_column_validation_edge_cases(graph: FeatureGraph):
    """Test edge cases for ID column validation in the joiner."""
    joiner = NarwhalsJoiner()

    # Test Case 1: Upstream has extra ID columns (should work)
    class UpstreamWithExtra(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["upstream_extra"]),
            deps=None,
            id_columns=["user_id", "session_id", "extra_id"],
        ),
    ):
        pass

    class TargetFewerColumns(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["target_fewer"]),
            deps=[FeatureDep(key=FeatureKey(["upstream_extra"]))],
            id_columns=["user_id", "session_id"],  # Doesn't require extra_id
        ),
    ):
        pass

    upstream_extra_data = nw.from_native(
        pl.DataFrame(
            {
                "user_id": [1, 2],
                "session_id": [10, 20],
                "extra_id": [100, 200],  # Extra column, not required by target
                "data_version": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                ],
            }
        ).lazy()
    )

    plan = graph.get_feature_plan(TargetFewerColumns.spec.key)

    # Should work because upstream has all required columns (and some extra)
    joined, _ = joiner.join_upstream(
        upstream_refs={"upstream_extra": upstream_extra_data},
        feature_spec=TargetFewerColumns.spec,
        feature_plan=plan,
    )
    result = joined.collect()
    assert len(result) == 2
    assert "user_id" in result.columns
    assert "session_id" in result.columns
    # extra_id may or may not be included (not required for join)

    # Test Case 2: Empty upstream refs (source feature) - should work
    class SourceWithCustomID(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["source"]),
            deps=None,
            id_columns=["entity_id"],
        ),
    ):
        pass

    plan2 = graph.get_feature_plan(SourceWithCustomID.spec.key)

    # No validation needed for empty upstream
    joined2, _ = joiner.join_upstream(
        upstream_refs={},
        feature_spec=SourceWithCustomID.spec,
        feature_plan=plan2,
    )
    result2 = joined2.collect()
    assert "entity_id" in result2.columns
    assert len(result2) == 0  # Empty source

    # Test Case 3: Multiple upstreams with different missing columns
    class Upstream1Missing(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["up1"]),
            deps=None,
            id_columns=["user_id"],  # Missing session_id
        ),
    ):
        pass

    class Upstream2Missing(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["up2"]),
            deps=None,
            id_columns=["session_id"],  # Missing user_id
        ),
    ):
        pass

    class TargetNeedsBoth(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["target_both"]),
            deps=[
                FeatureDep(key=FeatureKey(["up1"])),
                FeatureDep(key=FeatureKey(["up2"])),
            ],
            id_columns=["user_id", "session_id"],
        ),
    ):
        pass

    up1_data = nw.from_native(
        pl.DataFrame(
            {
                "user_id": [1, 2],
                # Missing session_id
                "data_version": [{"default": "h1"}, {"default": "h2"}],
            }
        ).lazy()
    )

    up2_data = nw.from_native(
        pl.DataFrame(
            {
                "session_id": [10, 20],
                # Missing user_id
                "data_version": [{"default": "h3"}, {"default": "h4"}],
            }
        ).lazy()
    )

    plan3 = graph.get_feature_plan(TargetNeedsBoth.spec.key)

    # Should fail with clear error about up1 missing session_id
    with pytest.raises(ValueError) as exc_info:
        joiner.join_upstream(
            upstream_refs={
                "up1": up1_data,
                "up2": up2_data,
            },
            feature_spec=TargetNeedsBoth.spec,
            feature_plan=plan3,
        )

    error_msg = str(exc_info.value)
    # The error should mention the first upstream with missing columns
    assert ("up1" in error_msg and "session_id" in error_msg) or (
        "up2" in error_msg and "user_id" in error_msg
    )
    assert "missing required ID columns" in error_msg

    # Test Case 4: Proper error message formatting
    class SingleMissing(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["single"]),
            deps=None,
            id_columns=["id1"],
        ),
    ):
        pass

    class TargetMultipleIDs(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["target_multi"]),
            deps=[FeatureDep(key=FeatureKey(["single"]))],
            id_columns=["id1", "id2", "id3"],  # Multiple required
        ),
    ):
        pass

    single_data = nw.from_native(
        pl.DataFrame(
            {
                "id1": [1, 2],
                # Missing id2 and id3
                "data_version": [{"default": "h1"}, {"default": "h2"}],
                "extra_col": ["a", "b"],
            }
        ).lazy()
    )

    plan4 = graph.get_feature_plan(TargetMultipleIDs.spec.key)

    with pytest.raises(ValueError) as exc_info:
        joiner.join_upstream(
            upstream_refs={"single": single_data},
            feature_spec=TargetMultipleIDs.spec,
            feature_plan=plan4,
        )

    error_msg = str(exc_info.value)
    # Should mention both missing columns
    assert "id2" in error_msg
    assert "id3" in error_msg
    assert "single" in error_msg  # Should mention which upstream
    assert "target feature requires" in error_msg.lower()


def test_backwards_compatibility_default_id_columns(graph: FeatureGraph):
    """Test that features without explicit id_columns still use sample_uid."""
    from metaxy.metadata_store import InMemoryMetadataStore

    # Create feature WITHOUT specifying id_columns (backwards compatibility)
    class LegacyFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["legacy"]),
            deps=None,
            fields=[FieldSpec(key=FieldKey(["data"]), code_version=1)],
            # No id_columns specified - should default to ["sample_uid"]
        ),
    ):
        pass

    # Verify id_columns() returns default
    assert LegacyFeature.id_columns() == ["sample_uid"]

    # Test with metadata store
    with InMemoryMetadataStore() as store:
        df = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "data_version": [
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
