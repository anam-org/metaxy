"""Tests for the three-component field provenance architecture."""

import narwhals as nw
import polars as pl
import pytest

from metaxy.data_versioning.calculators.polars import PolarsProvenanceByFieldCalculator
from metaxy.data_versioning.diff.narwhals import NarwhalsDiffResolver
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.data_versioning.joiners.narwhals import NarwhalsJoiner
from metaxy.models.feature import FeatureGraph, TestingFeature
from metaxy.models.feature_spec import FeatureDep, SampleFeatureSpec
from metaxy.models.field import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey


@pytest.fixture
def features(graph: FeatureGraph) -> dict[str, type[TestingFeature]]:
    """Create test features for component testing."""

    class UpstreamVideo(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video"]),
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),
            ],
        ),
    ):
        pass

    class UpstreamAudio(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["audio"]),
            fields=[
                FieldSpec(key=FieldKey(["waveform"]), code_version="1"),
            ],
        ),
    ):
        pass

    class ProcessedVideo(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["processed"]),
            deps=[FeatureDep(feature=FeatureKey(["video"]))],
            fields=[
                FieldSpec(key=FieldKey(["default"]), code_version="1"),
            ],
        ),
    ):
        pass

    class MultiUpstreamFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["multi"]),
            deps=[
                FeatureDep(feature=FeatureKey(["video"])),
                FeatureDep(feature=FeatureKey(["audio"])),
            ],
            fields=[
                FieldSpec(key=FieldKey(["fusion"]), code_version="1"),
                FieldSpec(key=FieldKey(["analysis"]), code_version="2"),
            ],
        ),
    ):
        pass

    return {
        "UpstreamVideo": UpstreamVideo,
        "UpstreamAudio": UpstreamAudio,
        "ProcessedVideo": ProcessedVideo,
        "MultiUpstreamFeature": MultiUpstreamFeature,
    }


def test_polars_joiner(features: dict[str, type[TestingFeature]], graph: FeatureGraph):
    """Test NarwhalsJoiner joins upstream features correctly."""
    joiner = NarwhalsJoiner()

    # Create upstream metadata
    video_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"frames": "hash_v1", "audio": "hash_a1"},
                    {"frames": "hash_v2", "audio": "hash_a2"},
                    {"frames": "hash_v3", "audio": "hash_a3"},
                ],
            }
        ).lazy()
    )

    upstream_refs = {"video": video_metadata}

    # Join upstream
    feature = features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)

    joined, mapping = joiner.join_upstream(
        upstream_refs=upstream_refs,
        feature_spec=feature.spec(),
        feature_plan=plan,
    )

    # Verify result
    result = joined.collect()
    assert len(result) == 3
    assert "sample_uid" in result.columns
    assert "video" in mapping
    assert mapping["video"] in result.columns

    # Verify provenance_by_field column was renamed
    assert "__upstream_video__metaxy_provenance_by_field" in result.columns


def test_polars_joiner_prefers_data_version(
    features: dict[str, type[TestingFeature]], graph: FeatureGraph
) -> None:
    """Joiner should prefer metaxy_data_version_by_field when available."""
    joiner = NarwhalsJoiner()

    video_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {"frames": "hash_v1", "audio": "hash_a1"},
                    {"frames": "hash_v2", "audio": "hash_a2"},
                ],
                "metaxy_data_version_by_field": [
                    {"frames": "override_v", "audio": "override_a"},
                    {"frames": "override_v", "audio": "override_a"},
                ],
            }
        ).lazy()
    )

    feature = features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)

    joined, mapping = joiner.join_upstream(
        upstream_refs={"video": video_metadata},
        feature_spec=feature.spec(),
        feature_plan=plan,
    )

    joined_pl = joined.collect()

    # Mapping should point to data_version column when available
    expected_column = "__upstream_video__metaxy_data_version_by_field"
    assert mapping["video"] == expected_column

    # Both provenance and data_version columns should be preserved
    assert expected_column in joined_pl.columns
    assert "__upstream_video__metaxy_provenance_by_field" in joined_pl.columns


def test_polars_hash_calculator(
    features: dict[str, type[TestingFeature]], graph: FeatureGraph
):
    """Test PolarsProvenanceByFieldCalculator computes hashes correctly."""
    calculator = PolarsProvenanceByFieldCalculator()

    # Verify supported algorithms
    assert HashAlgorithm.XXHASH64 in calculator.supported_algorithms
    assert calculator.default_algorithm == HashAlgorithm.XXHASH64

    # Create joined upstream data (output from joiner)
    joined_upstream = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "__upstream_video__metaxy_provenance_by_field": [
                    {"frames": "hash_v1", "audio": "hash_a1"},
                    {"frames": "hash_v2", "audio": "hash_a2"},
                ],
            }
        ).lazy()
    )

    upstream_mapping = {"video": "__upstream_video__metaxy_provenance_by_field"}

    # Calculate field provenances
    feature = features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)

    with_versions = calculator.calculate_provenance_by_field(
        joined_upstream=joined_upstream,
        feature_spec=feature.spec(),
        feature_plan=plan,
        upstream_column_mapping=upstream_mapping,
    )

    # Verify result
    result = with_versions.collect()
    assert "metaxy_provenance_by_field" in result.columns

    # Convert to Polars to check schema type
    result_pl = result.to_native()
    assert isinstance(result_pl.schema["metaxy_provenance_by_field"], pl.Struct)

    # Check provenance_by_field has 'default' field field
    provenance_sample = result["metaxy_provenance_by_field"][0]
    assert "default" in provenance_sample


def test_polars_calculator_uses_data_version_overrides(
    features: dict[str, type[TestingFeature]], graph: FeatureGraph
) -> None:
    """Calculator should hash using data_version overrides when provided."""
    calculator = PolarsProvenanceByFieldCalculator()
    joiner = NarwhalsJoiner()

    video_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"frames": "p1", "audio": "a1"},
                    {"frames": "p2", "audio": "a2"},
                    {"frames": "p3", "audio": "a3"},
                ],
                "metaxy_data_version_by_field": [
                    {"frames": "override_common", "audio": "override_common"},
                    {"frames": "override_common", "audio": "override_common"},
                    None,  # Fallback to provenance for this sample
                ],
            }
        ).lazy()
    )

    feature = features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)

    joined, mapping = joiner.join_upstream(
        upstream_refs={"video": video_metadata},
        feature_spec=feature.spec(),
        feature_plan=plan,
    )

    with_versions = calculator.calculate_provenance_by_field(
        joined_upstream=joined,
        feature_spec=feature.spec(),
        feature_plan=plan,
        upstream_column_mapping=mapping,
    ).collect()

    result_pl = with_versions.to_native()
    hashes = [row["default"] for row in result_pl["metaxy_provenance_by_field"]]

    # Rows with override should match
    assert hashes[0] == hashes[1]
    # Row without override should fall back to provenance and differ
    assert hashes[0] != hashes[2]


def test_update_detection_uses_provenance_not_data_version(
    features: dict[str, type[TestingFeature]], graph: FeatureGraph
) -> None:
    """Update detection should always use provenance_by_field, never data_version_by_field.

    This test verifies the key behavior:
    - data_version_by_field affects downstream computation (what they see)
    - provenance_by_field is used for update detection (what changed)

    Scenario:
    - Current metadata has both provenance and data_version (pinned to v1)
    - Target has new provenance (v2) but same data_version (still v1)
    - Result: Should detect NO changes because provenance matches
    """
    from metaxy.data_versioning.diff.narwhals import NarwhalsDiffResolver

    resolver = NarwhalsDiffResolver()

    # Current state: provenance=v2, data_version pinned to v1
    current_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"field_a": "prov_v2_1"},
                    {"field_a": "prov_v2_2"},
                    {"field_a": "prov_v2_3"},
                ],
                "metaxy_data_version_by_field": [
                    {"field_a": "pinned_v1"},  # Pinned version for downstream
                    {"field_a": "pinned_v1"},
                    {"field_a": "pinned_v1"},
                ],
            }
        ).lazy()
    )

    # Target state: Same provenance (v2), data_version still pinned to v1
    # No real change - provenance hasn't changed
    target_provenance = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"field_a": "prov_v2_1"},  # Same as current
                    {"field_a": "prov_v2_2"},
                    {"field_a": "prov_v2_3"},
                ],
            }
        ).lazy()
    )

    increment = resolver.find_changes(
        target_provenance=target_provenance,
        current_metadata=current_metadata,
        id_columns=["sample_uid"],
    )

    # No changes should be detected - provenance_by_field matches
    added = increment.added.collect()
    changed = increment.changed.collect()
    removed = increment.removed.collect()

    assert len(added) == 0, "Should not detect additions when provenance matches"
    assert len(changed) == 0, "Should not detect changes when provenance matches"
    assert len(removed) == 0, "Should not detect removals"

    # Now test the opposite: provenance changed but data_version stayed same
    # This SHOULD trigger update detection
    target_provenance_changed = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"field_a": "prov_v3_1"},  # Different from current
                    {"field_a": "prov_v3_2"},
                    {"field_a": "prov_v3_3"},
                ],
            }
        ).lazy()
    )

    increment_changed = resolver.find_changes(
        target_provenance=target_provenance_changed,
        current_metadata=current_metadata,
        id_columns=["sample_uid"],
    )

    changed_rows = increment_changed.changed.collect()
    assert len(changed_rows) == 3, (
        "Should detect all 3 rows as changed when provenance differs"
    )


def test_polars_hash_calculator_algorithms(
    features: dict[str, type[TestingFeature]], graph: FeatureGraph
):
    """Test different hash algorithms produce different results."""
    joined_upstream = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1],
                "__upstream_video__metaxy_provenance_by_field": [
                    {"frames": "hash_v1", "audio": "hash_a1"}
                ],
            }
        ).lazy()
    )

    upstream_mapping = {"video": "__upstream_video__metaxy_provenance_by_field"}
    feature = features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)

    # Calculate with xxHash64
    calc_xxhash = PolarsProvenanceByFieldCalculator()
    result_xxhash = calc_xxhash.calculate_provenance_by_field(
        joined_upstream=joined_upstream,
        feature_spec=feature.spec(),
        feature_plan=plan,
        upstream_column_mapping=upstream_mapping,
        hash_algorithm=HashAlgorithm.XXHASH64,
    ).collect()

    # Calculate with wyhash
    calc_wyhash = PolarsProvenanceByFieldCalculator()
    result_wyhash = calc_wyhash.calculate_provenance_by_field(
        joined_upstream=joined_upstream,
        feature_spec=feature.spec(),
        feature_plan=plan,
        upstream_column_mapping=upstream_mapping,
        hash_algorithm=HashAlgorithm.WYHASH,
    ).collect()

    # Different algorithms should produce different hashes
    hash_xxhash = result_xxhash["metaxy_provenance_by_field"][0]["default"]
    hash_wyhash = result_wyhash["metaxy_provenance_by_field"][0]["default"]
    assert hash_xxhash != hash_wyhash


def test_polars_diff_resolver_no_current() -> None:
    """Test diff resolver when no current metadata exists."""
    diff_resolver = NarwhalsDiffResolver()

    target_provenance = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                ],
            }
        ).lazy()
    )

    # No current metadata - all rows are added
    result = diff_resolver.find_changes(
        target_provenance=target_provenance,
        current_metadata=None,
        id_columns=["sample_uid"],  # Using default ID columns for testing
    )

    # Materialize lazy frames to check lengths
    assert len(result.added.collect()) == 3
    assert len(result.changed.collect()) == 0
    assert len(result.removed.collect()) == 0


def test_polars_diff_resolver_with_changes() -> None:
    """Test diff resolver identifies added, changed, and removed rows."""
    diff_resolver = NarwhalsDiffResolver()

    target_provenance = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3, 4],
                "metaxy_provenance_by_field": [
                    {"default": "hash1"},  # Unchanged
                    {"default": "hash2_new"},  # Changed
                    {"default": "hash3"},  # Unchanged
                    {"default": "hash4"},  # Added
                ],
            }
        ).lazy()
    )

    current_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3, 5],
                "metaxy_provenance_by_field": [
                    {"default": "hash1"},  # Same
                    {"default": "hash2_old"},  # Different
                    {"default": "hash3"},  # Same
                    {"default": "hash5"},  # Removed (not in target)
                ],
            }
        ).lazy()
    )

    result = diff_resolver.find_changes(
        target_provenance=target_provenance,
        current_metadata=current_metadata,
        id_columns=["sample_uid"],  # Using default ID columns for testing
    )

    # Added: sample_uid=4 - materialize to check
    added_df = result.added.collect()
    assert len(added_df) == 1
    assert added_df["sample_uid"][0] == 4

    # Changed: sample_uid=2
    changed_df = result.changed.collect()
    assert len(changed_df) == 1
    assert changed_df["sample_uid"][0] == 2

    # Removed: sample_uid=5
    removed_df = result.removed.collect()
    assert len(removed_df) == 1
    assert removed_df["sample_uid"][0] == 5


def test_full_pipeline_integration(
    features: dict[str, type[TestingFeature]], graph: FeatureGraph
):
    """Test all three components working together."""
    # Step 1: Join upstream
    joiner = NarwhalsJoiner()

    video_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"frames": "v1", "audio": "a1"},
                    {"frames": "v2", "audio": "a2"},
                    {"frames": "v3", "audio": "a3"},
                ],
            }
        ).lazy()
    )

    feature = features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)

    joined, mapping = joiner.join_upstream(
        upstream_refs={"video": video_metadata},
        feature_spec=feature.spec(),
        feature_plan=plan,
    )

    # Step 2: Calculate field provenances
    calculator = PolarsProvenanceByFieldCalculator()

    with_versions = calculator.calculate_provenance_by_field(
        joined_upstream=joined,
        feature_spec=feature.spec(),
        feature_plan=plan,
        upstream_column_mapping=mapping,
    )

    # Step 3: Diff with current
    diff_resolver = NarwhalsDiffResolver()

    # Simulate some current metadata
    current = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {"default": "old_hash1"},
                    {"default": "old_hash2"},
                ],
            }
        ).lazy()
    )

    diff_result = diff_resolver.find_changes(
        target_provenance=with_versions,
        current_metadata=current,
        id_columns=["sample_uid"],  # Using default ID columns for testing
    )

    # Added: sample_uid=3 (not in current) - materialize to check
    added = diff_result.added.collect()
    assert len(added) == 1
    assert added["sample_uid"][0] == 3

    # Changed: sample_uids 1, 2 (different hashes)
    changed = diff_result.changed.collect()
    assert len(changed) == 2
    assert set(changed["sample_uid"].to_list()) == {1, 2}

    # Removed: none (all current samples are in target)
    removed = diff_result.removed.collect()
    assert len(removed) == 0


# ========== Multi-Upstream Tests ==========


def test_polars_joiner_multiple_upstream(
    features: dict[str, type[TestingFeature]], graph: FeatureGraph
):
    """Test joining multiple upstream features."""
    joiner = NarwhalsJoiner()

    video_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"frames": "v1", "audio": "a1"},
                    {"frames": "v2", "audio": "a2"},
                    {"frames": "v3", "audio": "a3"},
                ],
            }
        ).lazy()
    )

    audio_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"waveform": "w1"},
                    {"waveform": "w2"},
                    {"waveform": "w3"},
                ],
            }
        ).lazy()
    )

    upstream_refs = {"video": video_metadata, "audio": audio_metadata}

    feature = features["MultiUpstreamFeature"]
    plan = graph.get_feature_plan(feature.spec().key)

    joined, mapping = joiner.join_upstream(
        upstream_refs=upstream_refs,
        feature_spec=feature.spec(),
        feature_plan=plan,
    )

    result = joined.collect()

    # Should have both upstream provenance_by_field columns
    assert "video" in mapping
    assert "audio" in mapping
    assert mapping["video"] in result.columns
    assert mapping["audio"] in result.columns

    # Should have all 3 samples (inner join on matching sample_uids)
    assert len(result) == 3


def test_polars_joiner_partial_overlap(graph: FeatureGraph) -> None:
    """Test joiner with partial sample_uid overlap (inner join behavior)."""
    joiner = NarwhalsJoiner()

    # Video has samples 1, 2, 3
    video_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"frames": "v1"},
                    {"frames": "v2"},
                    {"frames": "v3"},
                ],
            }
        ).lazy()
    )

    # Audio has samples 2, 3, 4
    audio_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [2, 3, 4],
                "metaxy_provenance_by_field": [
                    {"waveform": "w2"},
                    {"waveform": "w3"},
                    {"waveform": "w4"},
                ],
            }
        ).lazy()
    )

    # Create a simple feature with both deps

    class TestFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test"]),
            deps=[
                FeatureDep(feature=FeatureKey(["video"])),
                FeatureDep(feature=FeatureKey(["audio"])),
            ],
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    # Need to register upstream features for the plan to work
    class Video(
        TestingFeature,
        spec=SampleFeatureSpec(key=FeatureKey(["video"])),
    ):
        pass

    class Audio(
        TestingFeature,
        spec=SampleFeatureSpec(key=FeatureKey(["audio"])),
    ):
        pass

    plan = graph.get_feature_plan(TestFeature.spec().key)

    joined, _ = joiner.join_upstream(
        upstream_refs={"video": video_metadata, "audio": audio_metadata},
        feature_spec=TestFeature.spec(),
        feature_plan=plan,
    )

    result = joined.collect()

    # Inner join - only samples 2, 3 (present in BOTH)
    assert len(result) == 2
    assert set(result["sample_uid"].to_list()) == {2, 3}


# ========== Multi-Field Calculator Tests ==========


def test_polars_calculator_multiple_fields(
    features: dict[str, type[TestingFeature]], graph: FeatureGraph
):
    """Test calculator with multiple fields."""
    calculator = PolarsProvenanceByFieldCalculator()

    joined_upstream = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "__upstream_video__metaxy_provenance_by_field": [
                    {"frames": "v1", "audio": "a1"},
                    {"frames": "v2", "audio": "a2"},
                ],
                "__upstream_audio__metaxy_provenance_by_field": [
                    {"waveform": "w1"},
                    {"waveform": "w2"},
                ],
            }
        ).lazy()
    )

    upstream_mapping = {
        "video": "__upstream_video__metaxy_provenance_by_field",
        "audio": "__upstream_audio__metaxy_provenance_by_field",
    }

    feature = features["MultiUpstreamFeature"]
    plan = graph.get_feature_plan(feature.spec().key)

    with_versions = calculator.calculate_provenance_by_field(
        joined_upstream=joined_upstream,
        feature_spec=feature.spec(),
        feature_plan=plan,
        upstream_column_mapping=upstream_mapping,
    )

    result = with_versions.collect()

    # Should have provenance_by_field struct with both fields
    # Convert to Polars to access struct fields
    result_pl = result.to_native()
    provenance_schema = result_pl.schema["metaxy_provenance_by_field"]
    field_names = {f.name for f in provenance_schema.fields}
    assert field_names == {"fusion", "analysis"}

    # Different code versions should produce different hashes
    sample_dv = result["metaxy_provenance_by_field"][0]
    assert sample_dv["fusion"] != sample_dv["analysis"]


def test_polars_calculator_unsupported_algorithm(
    features: dict[str, type[TestingFeature]], graph: FeatureGraph
):
    """Test error when using unsupported hash algorithm."""
    calculator = PolarsProvenanceByFieldCalculator()

    joined_upstream = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1],
                "__upstream_video__metaxy_provenance_by_field": [{"frames": "v1"}],
            }
        ).lazy()
    )

    feature = features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)

    # Try to use an algorithm by creating a fake enum value
    class FakeAlgorithm:
        pass

    with pytest.raises(ValueError, match="not supported"):
        calculator.calculate_provenance_by_field(
            joined_upstream=joined_upstream,
            feature_spec=feature.spec(),
            feature_plan=plan,
            upstream_column_mapping={
                "video": "__upstream_video__metaxy_provenance_by_field"
            },
            hash_algorithm=FakeAlgorithm(),  # pyright: ignore[reportArgumentType]
        )


# ========== Edge Cases ==========


def test_diff_resolver_all_unchanged() -> None:
    """Test diff when all rows are unchanged."""
    diff_resolver = NarwhalsDiffResolver()

    target_provenance = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                ],
            }
        ).lazy()
    )

    # Current has same field_provenance
    current_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                ],
            }
        ).lazy()
    )

    result = diff_resolver.find_changes(
        target_provenance=target_provenance,
        current_metadata=current_metadata,
        id_columns=["sample_uid"],  # Using default ID columns for testing
    )

    # Nothing changed - materialize lazy frames to check
    assert len(result.added.collect()) == 0
    assert len(result.changed.collect()) == 0
    assert len(result.removed.collect()) == 0


def test_joiner_deterministic_order(
    features: dict[str, type[TestingFeature]], graph: FeatureGraph
):
    """Test that join order doesn't affect result."""
    joiner = NarwhalsJoiner()

    video_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [{"frames": "v1"}, {"frames": "v2"}],
            }
        ).lazy()
    )

    audio_metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [{"waveform": "w1"}, {"waveform": "w2"}],
            }
        ).lazy()
    )

    feature = features["MultiUpstreamFeature"]
    plan = graph.get_feature_plan(feature.spec().key)

    # Join with different key orders
    joined1, mapping1 = joiner.join_upstream(
        upstream_refs={"video": video_metadata, "audio": audio_metadata},
        feature_spec=feature.spec(),
        feature_plan=plan,
    )

    joined2, mapping2 = joiner.join_upstream(
        upstream_refs={"audio": audio_metadata, "video": video_metadata},
        feature_spec=feature.spec(),
        feature_plan=plan,
    )

    # Mappings should be the same
    assert mapping1 == mapping2

    # Results should be identical (order-independent)
    result1 = joined1.collect().sort("sample_uid")
    result2 = joined2.collect().sort("sample_uid")

    # Convert to Polars for comparison (Narwhals doesn't have equals method)
    assert result1.to_native().equals(result2.to_native())


# ========== Feature Method Override Tests ==========


def test_feature_join_upstream_override(graph: FeatureGraph):
    """Test Feature.load_input can be overridden."""

    class CustomJoinFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["custom"]),
            deps=[FeatureDep(feature=FeatureKey(["video"]))],
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        @classmethod
        def load_input(cls, joiner, upstream_refs):
            # Custom: delegate to joiner but it's overridden
            from metaxy.models.feature_spec import FeatureDep

            # Extract columns and renames from deps (custom logic)
            upstream_columns = {}
            upstream_renames = {}

            deps = cls.spec().deps
            if deps:
                for dep in deps:
                    if isinstance(dep, FeatureDep):
                        dep_key_str = dep.feature.to_string()
                        upstream_columns[dep_key_str] = dep.columns
                        upstream_renames[dep_key_str] = dep.rename

            plan = cls.graph.get_feature_plan(cls.spec().key)
            joined, mapping = joiner.join_upstream(
                upstream_refs=upstream_refs,
                feature_spec=cls.spec(),
                feature_plan=plan,
                upstream_columns=upstream_columns,
                upstream_renames=upstream_renames,
            )
            # Could add custom logic here
            return joined, mapping

    # Register upstream
    class Video(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video"]),
            fields=[FieldSpec(key=FieldKey(["frames"]), code_version="1")],
        ),
    ):
        pass

    joiner = NarwhalsJoiner()
    video_metadata = nw.from_native(
        pl.DataFrame(
            {"sample_uid": [1], "metaxy_provenance_by_field": [{"frames": "v1"}]}
        ).lazy()
    )

    # Call the overridden method
    joined, mapping = CustomJoinFeature.load_input(
        joiner=joiner,
        upstream_refs={"video": video_metadata},
    )

    result = joined.collect()
    assert len(result) == 1


def test_feature_resolve_diff_override(graph: FeatureGraph):
    """Test Feature.resolve_provenance_diff can be overridden."""

    class CustomDiffFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["custom_diff"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        @classmethod
        def resolve_provenance_diff(
            cls,
            diff_resolver,
            target_provenance,
            current_metadata,
            *,
            lazy=False,
        ):
            # Custom: call standard diff, then could modify results
            lazy_result = diff_resolver.find_changes(
                target_provenance,
                current_metadata,
                id_columns=cls.spec().id_columns,  # Pass ID columns from feature spec
            )
            # Could filter/modify result here

            # Materialize if lazy=False
            if not lazy:
                from metaxy.data_versioning.diff import Increment

                return Increment(
                    added=lazy_result.added.collect(),
                    changed=lazy_result.changed.collect(),
                    removed=lazy_result.removed.collect(),
                )
            return lazy_result

    diff_resolver = NarwhalsDiffResolver()

    target = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {"default": "new1"},
                    {"default": "new2"},
                ],
            }
        ).lazy()
    )

    current = nw.from_native(
        pl.DataFrame(
            {"sample_uid": [1], "metaxy_provenance_by_field": [{"default": "old1"}]}
        ).lazy()
    )

    # Call overridden method (this calls find_changes which needs current_feature_version=False)
    # The test feature class needs to pass the parameter through
    # Explicitly pass lazy=False to get Increment (eager) instead of LazyIncrement
    result = CustomDiffFeature.resolve_provenance_diff(
        diff_resolver=diff_resolver,
        target_provenance=target,
        current_metadata=current,
        lazy=False,
    )

    # Should identify: added=1, changed=1
    # Type assertion to help type checker understand result is Increment (eager)
    from metaxy.data_versioning.diff import Increment

    assert isinstance(result, Increment)
    assert len(result.added) == 1
    assert len(result.changed) == 1


# ========== Snapshot Tests ==========


def test_hash_output_snapshots(
    snapshot, features: dict[str, type[TestingFeature]], graph: FeatureGraph
):
    """Snapshot test to detect hash algorithm changes."""
    calculator = PolarsProvenanceByFieldCalculator()

    joined_upstream = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "__upstream_video__metaxy_provenance_by_field": [
                    {"frames": "frame_hash_1", "audio": "audio_hash_1"},
                    {"frames": "frame_hash_2", "audio": "audio_hash_2"},
                    {"frames": "frame_hash_3", "audio": "audio_hash_3"},
                ],
            }
        ).lazy()
    )

    upstream_mapping = {"video": "__upstream_video__metaxy_provenance_by_field"}
    feature = features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)

    with_versions = calculator.calculate_provenance_by_field(
        joined_upstream=joined_upstream,
        feature_spec=feature.spec(),
        feature_plan=plan,
        upstream_column_mapping=upstream_mapping,
        hash_algorithm=HashAlgorithm.XXHASH64,
    )

    result = with_versions.collect()
    hashes = result["metaxy_provenance_by_field"].struct.field("default").to_list()

    # Snapshot the hashes to detect algorithm changes
    assert hashes == snapshot


def test_multi_field_hash_snapshots(
    snapshot, features: dict[str, type[TestingFeature]], graph: FeatureGraph
):
    """Snapshot test for multi-field hash outputs."""
    calculator = PolarsProvenanceByFieldCalculator()

    joined_upstream = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1],
                "__upstream_video__metaxy_provenance_by_field": [
                    {"frames": "v1", "audio": "a1"}
                ],
                "__upstream_audio__metaxy_provenance_by_field": [{"waveform": "w1"}],
            }
        ).lazy()
    )

    upstream_mapping = {
        "video": "__upstream_video__metaxy_provenance_by_field",
        "audio": "__upstream_audio__metaxy_provenance_by_field",
    }

    feature = features["MultiUpstreamFeature"]
    plan = graph.get_feature_plan(feature.spec().key)

    with_versions = calculator.calculate_provenance_by_field(
        joined_upstream=joined_upstream,
        feature_spec=feature.spec(),
        feature_plan=plan,
        upstream_column_mapping=upstream_mapping,
    )

    result = with_versions.collect()
    provenance_by_field = result["metaxy_provenance_by_field"][0]

    # Snapshot both field hashes
    field_hashes = {
        "fusion": provenance_by_field["fusion"],
        "analysis": provenance_by_field["analysis"],
    }

    assert field_hashes == snapshot


# ========== Comprehensive Snapshot Tests ==========


@pytest.mark.parametrize(
    "hash_algorithm",
    [
        HashAlgorithm.XXHASH64,
        HashAlgorithm.XXHASH32,
        HashAlgorithm.WYHASH,
        HashAlgorithm.SHA256,
        HashAlgorithm.MD5,
    ],
)
def test_single_upstream_single_field_snapshots(
    snapshot,
    features: dict[str, type[TestingFeature]],
    graph: FeatureGraph,
    hash_algorithm,
):
    """Snapshot field provenances for single upstream, single field."""
    calculator = PolarsProvenanceByFieldCalculator()

    joined_upstream = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "__upstream_video__metaxy_provenance_by_field": [
                    {"frames": "v1", "audio": "a1"},
                    {"frames": "v2", "audio": "a2"},
                    {"frames": "v3", "audio": "a3"},
                ],
            }
        ).lazy()
    )

    feature = features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)

    with_versions = calculator.calculate_provenance_by_field(
        joined_upstream=joined_upstream,
        feature_spec=feature.spec(),
        feature_plan=plan,
        upstream_column_mapping={
            "video": "__upstream_video__metaxy_provenance_by_field"
        },
        hash_algorithm=hash_algorithm,
    )

    result = with_versions.collect()
    hashes = result["metaxy_provenance_by_field"].struct.field("default").to_list()

    assert hashes == snapshot


@pytest.mark.parametrize(
    "hash_algorithm",
    [
        HashAlgorithm.XXHASH64,
        HashAlgorithm.WYHASH,
        HashAlgorithm.SHA256,
    ],
)
def test_multi_upstream_multi_field_snapshots(
    snapshot,
    features: dict[str, type[TestingFeature]],
    graph: FeatureGraph,
    hash_algorithm,
):
    """Snapshot field provenances for multiple upstreams and fields."""
    calculator = PolarsProvenanceByFieldCalculator()

    joined_upstream = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "__upstream_video__metaxy_provenance_by_field": [
                    {"frames": "frame1", "audio": "audio1"},
                    {"frames": "frame2", "audio": "audio2"},
                ],
                "__upstream_audio__metaxy_provenance_by_field": [
                    {"waveform": "wave1"},
                    {"waveform": "wave2"},
                ],
            }
        ).lazy()
    )

    upstream_mapping = {
        "video": "__upstream_video__metaxy_provenance_by_field",
        "audio": "__upstream_audio__metaxy_provenance_by_field",
    }

    feature = features["MultiUpstreamFeature"]
    plan = graph.get_feature_plan(feature.spec().key)

    with_versions = calculator.calculate_provenance_by_field(
        joined_upstream=joined_upstream,
        feature_spec=feature.spec(),
        feature_plan=plan,
        upstream_column_mapping=upstream_mapping,
        hash_algorithm=hash_algorithm,
    )

    result = with_versions.collect()

    # Snapshot both field hashes for both samples
    field_provenance = []
    for i in range(len(result)):
        dv = result["metaxy_provenance_by_field"][i]
        field_provenance.append(
            {
                "sample_uid": result["sample_uid"][i],
                "fusion": dv["fusion"],
                "analysis": dv["analysis"],
            }
        )

    assert field_provenance == snapshot


def test_code_version_changes_snapshots(snapshot, graph: FeatureGraph):
    """Snapshot showing code version changes produce different hashes."""
    calculator = PolarsProvenanceByFieldCalculator()

    joined_upstream = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1],
                "__upstream_video__metaxy_provenance_by_field": [{"frames": "v1"}],
            }
        ).lazy()
    )

    upstream_mapping = {"video": "__upstream_video__metaxy_provenance_by_field"}

    # Same feature, different code versions
    versions_by_code_version = {}

    for code_version in [1, 2, 5, 10]:

        class TestFeature(
            TestingFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey([f"test_v{code_version}"]),
                deps=[FeatureDep(feature=FeatureKey(["video"]))],
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version=str(code_version))
                ],
            ),
        ):
            pass

        # Register video upstream
        if code_version == 1:

            class Video(
                TestingFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["video"]),
                    fields=[FieldSpec(key=FieldKey(["frames"]), code_version="1")],
                ),
            ):
                pass

        plan = graph.get_feature_plan(TestFeature.spec().key)

        with_versions = calculator.calculate_provenance_by_field(
            joined_upstream=joined_upstream,
            feature_spec=TestFeature.spec(),
            feature_plan=plan,
            upstream_column_mapping=upstream_mapping,
        )

        result = with_versions.collect()
        hash_value = result["metaxy_provenance_by_field"][0]["default"]
        versions_by_code_version[f"code_v{code_version}"] = hash_value

    # Different code versions should produce different hashes
    unique_hashes = set(versions_by_code_version.values())
    assert len(unique_hashes) == 4, "All code versions should produce unique hashes"

    assert versions_by_code_version == snapshot


def test_upstream_data_changes_snapshots(snapshot, graph: FeatureGraph):
    """Snapshot showing upstream data changes produce different hashes."""
    calculator = PolarsProvenanceByFieldCalculator()

    class Video(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video"]),
            fields=[FieldSpec(key=FieldKey(["frames"]), code_version="1")],
        ),
    ):
        pass

    class Processed(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["processed"]),
            deps=[FeatureDep(feature=FeatureKey(["video"]))],
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    plan = graph.get_feature_plan(Processed.spec().key)

    # Different upstream data scenarios
    scenarios = {
        "original": [{"frames": "hash_original"}],
        "modified": [{"frames": "hash_modified"}],
        "different": [{"frames": "hash_completely_different"}],
    }

    hashes_by_scenario = {}

    for scenario_name, provenance_list in scenarios.items():
        joined_upstream = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1],
                    "__upstream_video__metaxy_provenance_by_field": provenance_list,
                }
            ).lazy()
        )

        with_versions = calculator.calculate_provenance_by_field(
            joined_upstream=joined_upstream,
            feature_spec=Processed.spec(),
            feature_plan=plan,
            upstream_column_mapping={
                "video": "__upstream_video__metaxy_provenance_by_field"
            },
        )

        result = with_versions.collect()
        hashes_by_scenario[scenario_name] = result["metaxy_provenance_by_field"][0][
            "default"
        ]

    # All scenarios should produce different hashes
    unique_hashes = set(hashes_by_scenario.values())
    assert len(unique_hashes) == 3, (
        "Different upstream data should produce different hashes"
    )

    assert hashes_by_scenario == snapshot


def test_diff_result_snapshots(snapshot):
    """Snapshot the structure of Increment for various scenarios."""
    diff_resolver = NarwhalsDiffResolver()

    target = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3, 4, 5],
                "metaxy_provenance_by_field": [
                    {"default": "unchanged_hash"},
                    {"default": "changed_new_hash"},
                    {"default": "unchanged_hash2"},
                    {"default": "new_sample_hash"},
                    {"default": "another_new_hash"},
                ],
            }
        ).lazy()
    )

    current = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3, 6],
                "metaxy_provenance_by_field": [
                    {"default": "unchanged_hash"},
                    {"default": "changed_old_hash"},
                    {"default": "unchanged_hash2"},
                    {"default": "removed_sample_hash"},
                ],
            }
        ).lazy()
    )

    result = diff_resolver.find_changes(
        target_provenance=target,
        current_metadata=current,
        id_columns=["sample_uid"],  # Using default ID columns for testing
    )

    # Snapshot the sample_uids in each category - materialize lazy frames first
    diff_summary = {
        "added_ids": sorted(result.added.collect()["sample_uid"].to_list()),
        "changed_ids": sorted(result.changed.collect()["sample_uid"].to_list()),
        "removed_ids": sorted(result.removed.collect()["sample_uid"].to_list()),
    }

    assert diff_summary == snapshot
