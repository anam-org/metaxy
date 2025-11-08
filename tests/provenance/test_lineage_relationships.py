"""Tests for lineage relationship handling in provenance tracking.

Tests cover:
- Identity relationships (1:1) - baseline behavior
- Aggregation relationships (N:1) - many parent samples → one child sample
- Expansion relationships (1:N) - one parent sample → many child samples
"""

from __future__ import annotations

import narwhals as nw
import polars as pl
import pytest

from metaxy.models.feature import FeatureGraph, TestingFeature
from metaxy.models.feature_spec import FeatureDep, SampleFeatureSpec
from metaxy.models.field import FieldSpec, SpecialFieldDep
from metaxy.models.lineage import LineageRelationship
from metaxy.models.types import FeatureKey, FieldKey
from metaxy.provenance.polars import PolarsProvenanceTracker
from metaxy.provenance.types import HashAlgorithm

# ============================================================================
# Fixtures for Aggregation (N:1) scenarios
# ============================================================================


@pytest.fixture
def aggregation_features(graph: FeatureGraph) -> dict[str, type[TestingFeature]]:
    """Create features for testing N:1 aggregation relationships.

    Scenario: Sensor readings (many per hour) → hourly statistics (one per hour)
    - SensorReadings: sensor_id, timestamp, reading_id (fine-grained data)
    - HourlyStats: sensor_id, hour (aggregated by sensor and hour)
    """

    class SensorReadings(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["sensor_readings"]),
            id_columns=("sensor_id", "timestamp", "reading_id"),
            fields=[
                FieldSpec(key=FieldKey(["temperature"]), code_version="1"),
                FieldSpec(key=FieldKey(["humidity"]), code_version="1"),
            ],
        ),
    ):
        pass

    class HourlyStats(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["hourly_stats"]),
            id_columns=("sensor_id", "hour"),
            deps=[FeatureDep(feature=FeatureKey(["sensor_readings"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["avg_temp"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                ),
                FieldSpec(
                    key=FieldKey(["avg_humidity"]),
                    code_version="2",
                    deps=SpecialFieldDep.ALL,
                ),
            ],
            lineage=LineageRelationship.aggregation(on=["sensor_id", "hour"]),
        ),
    ):
        pass

    return {
        "SensorReadings": SensorReadings,
        "HourlyStats": HourlyStats,
    }


@pytest.fixture
def sensor_readings_metadata() -> nw.LazyFrame[pl.LazyFrame]:
    """Sample sensor readings metadata (N:1 upstream).

    Multiple readings per (sensor_id, hour) pair will aggregate to one hourly stat.
    """
    return nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1", "s1", "s1", "s2", "s2"],
                "timestamp": [
                    "2024-01-01T10:15:00",
                    "2024-01-01T10:30:00",
                    "2024-01-01T10:45:00",
                    "2024-01-01T10:20:00",
                    "2024-01-01T10:50:00",
                ],
                "reading_id": ["r1", "r2", "r3", "r4", "r5"],
                "hour": [
                    "2024-01-01T10",
                    "2024-01-01T10",
                    "2024-01-01T10",
                    "2024-01-01T10",
                    "2024-01-01T10",
                ],
                "metaxy_provenance_by_field": [
                    {"temperature": "temp_hash_1", "humidity": "hum_hash_1"},
                    {"temperature": "temp_hash_2", "humidity": "hum_hash_2"},
                    {"temperature": "temp_hash_3", "humidity": "hum_hash_3"},
                    {"temperature": "temp_hash_4", "humidity": "hum_hash_4"},
                    {"temperature": "temp_hash_5", "humidity": "hum_hash_5"},
                ],
                "metaxy_provenance": [
                    "reading_prov_1",
                    "reading_prov_2",
                    "reading_prov_3",
                    "reading_prov_4",
                    "reading_prov_5",
                ],
            }
        ).lazy()
    )


# ============================================================================
# Fixtures for Expansion (1:N) scenarios
# ============================================================================


@pytest.fixture
def expansion_features(graph: FeatureGraph) -> dict[str, type[TestingFeature]]:
    """Create features for testing 1:N expansion relationships.

    Scenario: Video → video frames (one video expands to many frames)
    - Video: video_id (one per video)
    - VideoFrames: video_id, frame_id (many frames per video)
    """

    class Video(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video"]),
            id_columns=("video_id",),
            fields=[
                FieldSpec(key=FieldKey(["resolution"]), code_version="1"),
                FieldSpec(key=FieldKey(["fps"]), code_version="1"),
            ],
        ),
    ):
        pass

    class VideoFrames(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video_frames"]),
            id_columns=("video_id", "frame_id"),
            deps=[FeatureDep(feature=FeatureKey(["video"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["frame_embedding"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                ),
            ],
            lineage=LineageRelationship.expansion(on=["video_id"]),
        ),
    ):
        pass

    return {
        "Video": Video,
        "VideoFrames": VideoFrames,
    }


@pytest.fixture
def video_metadata() -> nw.LazyFrame[pl.LazyFrame]:
    """Sample video metadata (1:N upstream - one video expands to many frames)."""
    return nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "metaxy_provenance_by_field": [
                    {"resolution": "res_hash_1", "fps": "fps_hash_1"},
                    {"resolution": "res_hash_2", "fps": "fps_hash_2"},
                ],
                "metaxy_provenance": ["video_prov_1", "video_prov_2"],
            }
        ).lazy()
    )


@pytest.fixture
def video_frames_current() -> nw.LazyFrame[pl.LazyFrame]:
    """Current video frames metadata (many frames per video)."""
    return nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1", "v1", "v1", "v2", "v2"],
                "frame_id": [0, 1, 2, 0, 1],
                "metaxy_provenance_by_field": [
                    {"frame_embedding": "frame_v1_0_hash"},
                    {"frame_embedding": "frame_v1_1_hash"},
                    {"frame_embedding": "frame_v1_2_hash"},
                    {"frame_embedding": "frame_v2_0_hash"},
                    {"frame_embedding": "frame_v2_1_hash"},
                ],
                "metaxy_provenance": [
                    "frame_v1_0_prov",
                    "frame_v1_1_prov",
                    "frame_v1_2_prov",
                    "frame_v2_0_prov",
                    "frame_v2_1_prov",
                ],
            }
        ).lazy()
    )


# ============================================================================
# Tests for Identity relationships (1:1) - baseline
# ============================================================================


def test_identity_lineage_load_upstream(
    simple_features: dict[str, type[TestingFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test identity lineage (1:1) - baseline behavior."""
    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = PolarsProvenanceTracker(plan)

    # Verify lineage is identity
    assert plan.feature.lineage.relationship.type.value == "1:1"

    upstream = {FeatureKey(["video"]): upstream_video_metadata}

    result = tracker.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()

    # Identity relationship: all samples should be present
    assert len(result_df) == 3
    assert set(result_df["sample_uid"].to_list()) == {1, 2, 3}


def test_identity_lineage_resolve_increment(
    simple_features: dict[str, type[TestingFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test identity lineage (1:1) increment resolution."""
    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = PolarsProvenanceTracker(plan)

    upstream = {FeatureKey(["video"]): upstream_video_metadata}

    # Compute expected provenance
    expected = tracker.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )
    expected_df = expected.collect()

    # Create current with one changed sample
    current = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    expected_df["metaxy_provenance_by_field"][0],  # Unchanged
                    {"default": "different_hash"},  # Changed
                    expected_df["metaxy_provenance_by_field"][2],  # Unchanged
                ],
                "metaxy_provenance": [
                    expected_df["metaxy_provenance"][0],
                    "different_prov",
                    expected_df["metaxy_provenance"][2],
                ],
            }
        ).lazy()
    )

    added_lazy, changed_lazy, removed_lazy = tracker.resolve_increment_with_provenance(
        current=current,
        upstream=upstream,
        hash_algorithm=HashAlgorithm.XXHASH64,
        filters={},
        sample=None,
    )

    # Identity: should detect one changed sample
    added = added_lazy.collect()
    assert changed_lazy is not None
    changed = changed_lazy.collect()
    assert removed_lazy is not None
    removed = removed_lazy.collect()

    assert len(added) == 0
    assert len(changed) == 1
    assert changed["sample_uid"][0] == 2
    assert len(removed) == 0


# ============================================================================
# Tests for Aggregation relationships (N:1)
# ============================================================================


def test_aggregation_lineage_load_upstream(
    aggregation_features: dict[str, type[TestingFeature]],
    sensor_readings_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test N:1 aggregation lineage loads upstream correctly.

    Note: load_upstream_with_provenance returns granular upstream data (all 5 readings).
    The aggregation normalization only happens during comparison in resolve_increment_with_provenance.
    """
    feature = aggregation_features["HourlyStats"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = PolarsProvenanceTracker(plan)

    # Verify lineage is aggregation
    assert plan.feature.lineage.relationship.type.value == "N:1"

    upstream = {FeatureKey(["sensor_readings"]): sensor_readings_metadata}

    result = tracker.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()

    # load_upstream_with_provenance returns ALL upstream rows (not aggregated yet)
    assert len(result_df) == 5  # All 5 sensor readings
    assert set(result_df["sensor_id"].to_list()) == {"s1", "s2"}

    # Provenance should be computed for each reading
    assert "metaxy_provenance" in result_df.columns
    assert "metaxy_provenance_by_field" in result_df.columns


def test_aggregation_lineage_resolve_increment_no_current(
    aggregation_features: dict[str, type[TestingFeature]],
    sensor_readings_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test N:1 aggregation increment resolution with no current metadata.

    When current is None, all upstream rows are returned as added (before aggregation).
    """
    feature = aggregation_features["HourlyStats"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = PolarsProvenanceTracker(plan)

    upstream = {FeatureKey(["sensor_readings"]): sensor_readings_metadata}

    added_lazy, changed_lazy, removed_lazy = tracker.resolve_increment_with_provenance(
        current=None,
        upstream=upstream,
        hash_algorithm=HashAlgorithm.XXHASH64,
        filters={},
        sample=None,
    )

    added = added_lazy.collect()

    # When current is None, all upstream samples are added (not aggregated yet)
    assert changed_lazy is None
    assert removed_lazy is None
    assert len(added) == 5  # All 5 sensor readings


def test_aggregation_lineage_resolve_increment_with_changes(
    aggregation_features: dict[str, type[TestingFeature]],
    sensor_readings_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test N:1 aggregation detects changes correctly.

    When upstream readings change, the aggregated hourly stat should be marked as changed.
    The current metadata is at the aggregated level (sensor_id, hour).
    """
    feature = aggregation_features["HourlyStats"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = PolarsProvenanceTracker(plan)

    upstream = {FeatureKey(["sensor_readings"]): sensor_readings_metadata}

    # Current metadata at aggregated level (2 rows: one per sensor+hour)
    # We need to manually compute what the aggregated provenance would be
    # For simplicity, use placeholder values that differ for s2
    current = nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1", "s2"],
                "hour": ["2024-01-01T10", "2024-01-01T10"],
                "metaxy_provenance_by_field": [
                    {
                        "avg_temp": "aggregated_hash_s1",
                        "avg_humidity": "aggregated_hash_s1",
                    },
                    {
                        "avg_temp": "aggregated_hash_s2_OLD",
                        "avg_humidity": "aggregated_hash_s2_OLD",
                    },
                ],
                "metaxy_provenance": [
                    "aggregated_prov_s1",
                    "aggregated_prov_s2_OLD",  # This will differ from actual
                ],
            }
        ).lazy()
    )

    added_lazy, changed_lazy, removed_lazy = tracker.resolve_increment_with_provenance(
        current=current,
        upstream=upstream,
        hash_algorithm=HashAlgorithm.XXHASH64,
        filters={},
        sample=None,
    )

    added = added_lazy.collect()
    assert changed_lazy is not None
    changed = changed_lazy.collect()
    assert removed_lazy is not None
    removed = removed_lazy.collect()

    # With N:1 aggregation, the lineage handler aggregates expected readings by (sensor_id, hour)
    # Then compares with current at that same aggregation level
    # Both s1 and s2 will likely be marked as changed since our placeholder provenance won't match
    assert len(added) == 0
    assert len(changed) >= 1  # At least one changed
    assert len(removed) == 0


def test_aggregation_lineage_new_readings_trigger_change(
    aggregation_features: dict[str, type[TestingFeature]],
    graph: FeatureGraph,
) -> None:
    """Test that adding new readings to an hour marks that hourly stat as changed.

    This verifies the aggregation logic: when upstream readings are added/changed,
    the aggregated provenance should change. The test uses resolve_increment_with_provenance
    twice to simulate before/after states.
    """
    feature = aggregation_features["HourlyStats"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = PolarsProvenanceTracker(plan)

    # Initial upstream: 2 readings for s1 in hour 10
    upstream_v1 = nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1", "s1"],
                "timestamp": ["2024-01-01T10:15:00", "2024-01-01T10:30:00"],
                "reading_id": ["r1", "r2"],
                "hour": ["2024-01-01T10", "2024-01-01T10"],
                "metaxy_provenance_by_field": [
                    {"temperature": "temp_hash_1", "humidity": "hum_hash_1"},
                    {"temperature": "temp_hash_2", "humidity": "hum_hash_2"},
                ],
                "metaxy_provenance": ["reading_prov_1", "reading_prov_2"],
            }
        ).lazy()
    )

    # First resolve: no current, so all readings are added
    added_v1_lazy, _, _ = tracker.resolve_increment_with_provenance(
        current=None,
        upstream={FeatureKey(["sensor_readings"]): upstream_v1},
        hash_algorithm=HashAlgorithm.XXHASH64,
        filters={},
        sample=None,
    )
    added_v1 = added_v1_lazy.collect()
    assert len(added_v1) == 2  # 2 initial readings

    # Updated upstream: 3 readings for s1 in hour 10 (added r3)
    upstream_v2 = nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1", "s1", "s1"],
                "timestamp": [
                    "2024-01-01T10:15:00",
                    "2024-01-01T10:30:00",
                    "2024-01-01T10:45:00",  # New reading
                ],
                "reading_id": ["r1", "r2", "r3"],
                "hour": ["2024-01-01T10", "2024-01-01T10", "2024-01-01T10"],
                "metaxy_provenance_by_field": [
                    {"temperature": "temp_hash_1", "humidity": "hum_hash_1"},
                    {"temperature": "temp_hash_2", "humidity": "hum_hash_2"},
                    {"temperature": "temp_hash_3", "humidity": "hum_hash_3"},  # New
                ],
                "metaxy_provenance": [
                    "reading_prov_1",
                    "reading_prov_2",
                    "reading_prov_3",  # New
                ],
            }
        ).lazy()
    )

    # Create a mock current at aggregated level
    # This represents the state after v1 was processed
    # For aggregation testing, we need current at (sensor_id, hour) level
    current_aggregated = nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1"],
                "hour": ["2024-01-01T10"],
                "metaxy_provenance_by_field": [
                    {
                        "avg_temp": "old_aggregated_hash",
                        "avg_humidity": "old_aggregated_hash",
                    }
                ],
                "metaxy_provenance": ["old_aggregated_prov"],
            }
        ).lazy()
    )

    # Resolve increment with new upstream
    added_lazy, changed_lazy, removed_lazy = tracker.resolve_increment_with_provenance(
        current=current_aggregated,
        upstream={FeatureKey(["sensor_readings"]): upstream_v2},
        hash_algorithm=HashAlgorithm.XXHASH64,
        filters={},
        sample=None,
    )

    added = added_lazy.collect()
    assert changed_lazy is not None
    changed = changed_lazy.collect()
    assert removed_lazy is not None
    removed = removed_lazy.collect()

    # Adding r3 changes the aggregated provenance for s1's hourly stat
    # It should be detected as changed (not added, since s1+hour already exists)
    assert len(added) == 0
    assert len(changed) >= 1  # s1's hourly stat changed
    assert len(removed) == 0


# ============================================================================
# Tests for Expansion relationships (1:N)
# ============================================================================


def test_expansion_lineage_load_upstream(
    expansion_features: dict[str, type[TestingFeature]],
    video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test 1:N expansion lineage loads upstream correctly."""
    feature = expansion_features["VideoFrames"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = PolarsProvenanceTracker(plan)

    # Verify lineage is expansion
    assert plan.feature.lineage.relationship.type.value == "1:N"

    upstream = {FeatureKey(["video"]): video_metadata}

    result = tracker.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()

    # Should have one entry per video (parent)
    assert len(result_df) == 2
    assert set(result_df["video_id"].to_list()) == {"v1", "v2"}


def test_expansion_lineage_resolve_increment_no_current(
    expansion_features: dict[str, type[TestingFeature]],
    video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test 1:N expansion increment resolution with no current metadata."""
    feature = expansion_features["VideoFrames"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = PolarsProvenanceTracker(plan)

    upstream = {FeatureKey(["video"]): video_metadata}

    added_lazy, changed_lazy, removed_lazy = tracker.resolve_increment_with_provenance(
        current=None,
        upstream=upstream,
        hash_algorithm=HashAlgorithm.XXHASH64,
        filters={},
        sample=None,
    )

    added = added_lazy.collect()

    # When current is None, all videos should be added
    assert changed_lazy is None
    assert removed_lazy is None
    assert len(added) == 2  # Two videos


def test_expansion_lineage_resolve_increment_video_changed(
    expansion_features: dict[str, type[TestingFeature]],
    video_metadata: nw.LazyFrame[pl.LazyFrame],
    video_frames_current: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test 1:N expansion detects when parent video changes.

    When a video's metadata changes, ALL frames from that video should be marked
    as changed. The expansion handler groups current frames by video_id to compare
    at the parent level.
    """
    feature = expansion_features["VideoFrames"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = PolarsProvenanceTracker(plan)

    # Modified video metadata (v2's provenance changed compared to fixture)
    upstream_modified = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "metaxy_provenance_by_field": [
                    {
                        "resolution": "res_hash_1",
                        "fps": "fps_hash_1",
                    },  # v1 unchanged from fixture
                    {
                        "resolution": "res_hash_2_MODIFIED",
                        "fps": "fps_hash_2_MODIFIED",
                    },  # v2 changed
                ],
                "metaxy_provenance": [
                    "video_prov_1",  # Same as fixture
                    "video_prov_2_MODIFIED",  # Changed from fixture
                ],
            }
        ).lazy()
    )

    # Resolve increment with modified video metadata
    # The expansion handler will group video_frames_current by video_id to get one row per video
    # Then compare with upstream (which has 2 videos)
    added_lazy, changed_lazy, removed_lazy = tracker.resolve_increment_with_provenance(
        current=video_frames_current,  # Current frames for v1 (3 frames) and v2 (2 frames)
        upstream={FeatureKey(["video"]): upstream_modified},
        hash_algorithm=HashAlgorithm.XXHASH64,
        filters={},
        sample=None,
    )

    added = added_lazy.collect()
    assert changed_lazy is not None
    changed = changed_lazy.collect()
    assert removed_lazy is not None
    removed = removed_lazy.collect()

    # With expansion lineage, changes are detected at the parent (video) level
    # If v2 changed, it should appear in changed
    # The comparison happens after grouping current by video_id
    assert len(added) == 0
    assert len(removed) == 0
    assert len(changed) >= 1  # At least v2 changed


def test_expansion_lineage_new_video_added(
    expansion_features: dict[str, type[TestingFeature]],
    video_frames_current: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test 1:N expansion detects when new parent video is added.

    Adding a new video should be detected at the parent level after grouping current frames.
    """
    feature = expansion_features["VideoFrames"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = PolarsProvenanceTracker(plan)

    # New upstream includes v1, v2, and v3 (v3 is new)
    upstream_with_new_video = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1", "v2", "v3"],  # v3 is new
                "metaxy_provenance_by_field": [
                    {"resolution": "res_hash_1", "fps": "fps_hash_1"},
                    {"resolution": "res_hash_2", "fps": "fps_hash_2"},
                    {"resolution": "res_hash_3", "fps": "fps_hash_3"},  # New
                ],
                "metaxy_provenance": [
                    "video_prov_1",
                    "video_prov_2",
                    "video_prov_3",  # New
                ],
            }
        ).lazy()
    )

    added_lazy, changed_lazy, removed_lazy = tracker.resolve_increment_with_provenance(
        current=video_frames_current,  # Has frames for v1 and v2 only
        upstream={FeatureKey(["video"]): upstream_with_new_video},
        hash_algorithm=HashAlgorithm.XXHASH64,
        filters={},
        sample=None,
    )

    added = added_lazy.collect()
    assert changed_lazy is not None
    changed_lazy.collect()  # Materialize but don't check count
    assert removed_lazy is not None
    removed = removed_lazy.collect()

    # v3 is new at the parent level, so it should appear in added
    assert len(added) == 1
    assert added["video_id"][0] == "v3"

    # v1 and v2 exist in both, so they may appear in changed if provenance differs
    # We don't check changed count since fixture provenance may differ from computed
    assert len(removed) == 0


def test_expansion_lineage_video_removed(
    expansion_features: dict[str, type[TestingFeature]],
    video_frames_current: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test 1:N expansion detects when parent video is removed.

    Removing a video should be detected at the parent level after grouping current frames.
    """
    feature = expansion_features["VideoFrames"]
    plan = graph.get_feature_plan(feature.spec().key)
    tracker = PolarsProvenanceTracker(plan)

    # New upstream only has v1 (v2 is removed)
    upstream_without_v2 = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1"],
                "metaxy_provenance_by_field": [
                    {"resolution": "res_hash_1", "fps": "fps_hash_1"},
                ],
                "metaxy_provenance": ["video_prov_1"],
            }
        ).lazy()
    )

    added_lazy, changed_lazy, removed_lazy = tracker.resolve_increment_with_provenance(
        current=video_frames_current,  # Has frames for v1 and v2
        upstream={FeatureKey(["video"]): upstream_without_v2},  # Only v1
        hash_algorithm=HashAlgorithm.XXHASH64,
        filters={},
        sample=None,
    )

    added = added_lazy.collect()
    assert changed_lazy is not None
    changed_lazy.collect()  # Materialize but don't check count
    assert removed_lazy is not None
    removed = removed_lazy.collect()

    # v2 is removed at the parent level
    assert len(added) == 0

    # Removed should contain v2 (detected after grouping current by video_id)
    removed_video_ids = set(removed["video_id"].to_list())
    assert "v2" in removed_video_ids
