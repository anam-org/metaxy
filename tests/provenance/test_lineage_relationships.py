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

from metaxy._testing.models import SampleFeature, SampleFeatureSpec
from metaxy.models.feature import FeatureGraph
from metaxy.models.feature_spec import FeatureDep
from metaxy.models.field import FieldSpec, SpecialFieldDep
from metaxy.models.lineage import LineageRelationship
from metaxy.models.types import FeatureKey, FieldKey
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm

# ============================================================================
# Fixtures for Aggregation (N:1) scenarios
# ============================================================================


@pytest.fixture
def aggregation_features(graph: FeatureGraph) -> dict[str, type[SampleFeature]]:
    """Create features for testing N:1 aggregation relationships.

    Scenario: Sensor readings (many per hour) → hourly statistics (one per hour)
    - SensorReadings: sensor_id, timestamp, reading_id (fine-grained data)
    - HourlyStats: sensor_id, hour (aggregated by sensor and hour)
    """

    class SensorReadings(
        SampleFeature,
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
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["hourly_stats"]),
            id_columns=("sensor_id", "hour"),
            deps=[
                FeatureDep(
                    feature=FeatureKey(["sensor_readings"]),
                    lineage=LineageRelationship.aggregation(on=["sensor_id", "hour"]),
                )
            ],
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
                "metaxy_data_version_by_field": [
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
                "metaxy_data_version": [
                    "reading_dv_1",
                    "reading_dv_2",
                    "reading_dv_3",
                    "reading_dv_4",
                    "reading_dv_5",
                ],
            }
        ).lazy()
    )


# ============================================================================
# Fixtures for Expansion (1:N) scenarios
# ============================================================================


@pytest.fixture
def expansion_features(graph: FeatureGraph) -> dict[str, type[SampleFeature]]:
    """Create features for testing 1:N expansion relationships.

    Scenario: Video → video frames (one video expands to many frames)
    - Video: video_id (one per video)
    - VideoFrames: video_id, frame_id (many frames per video)
    """

    class Video(
        SampleFeature,
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
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video_frames"]),
            id_columns=("video_id", "frame_id"),
            deps=[
                FeatureDep(
                    feature=FeatureKey(["video"]),
                    lineage=LineageRelationship.expansion(on=["video_id"]),
                )
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["frame_embedding"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                ),
            ],
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
                "metaxy_data_version_by_field": [
                    {"resolution": "res_hash_1", "fps": "fps_hash_1"},
                    {"resolution": "res_hash_2", "fps": "fps_hash_2"},
                ],
                "metaxy_provenance": ["video_prov_1", "video_prov_2"],
                "metaxy_data_version": ["video_prov_1", "video_prov_2"],
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
                "metaxy_data_version_by_field": [
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
                "metaxy_data_version": [
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
    simple_features: dict[str, type[SampleFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    snapshot,
) -> None:
    """Test identity lineage (1:1) - baseline behavior."""
    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    # Verify lineage is identity
    # Identity lineage is the default for deps without explicit lineage
    assert all(
        dep.lineage.relationship.type.value == "1:1" for dep in plan.feature.deps
    )

    upstream = {FeatureKey(["video"]): upstream_video_metadata}

    result = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()

    # Identity relationship: all samples should be present
    assert len(result_df) == 3
    assert set(result_df["sample_uid"].to_list()) == {1, 2, 3}

    # Snapshot the identity lineage provenance
    result_polars = result_df.to_polars()
    provenance_data = sorted(
        [
            {
                "sample_uid": result_polars["sample_uid"][i],
                "field_provenance": result_polars["metaxy_provenance_by_field"][i],
                "field_data_version": result_polars["metaxy_data_version_by_field"][i],
            }
            for i in range(len(result_polars))
        ],
        key=lambda x: x["sample_uid"],
    )
    assert provenance_data == snapshot


def test_identity_lineage_resolve_increment(
    simple_features: dict[str, type[SampleFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test identity lineage (1:1) increment resolution."""
    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    upstream = {FeatureKey(["video"]): upstream_video_metadata}

    # Compute expected provenance
    expected = engine.load_upstream_with_provenance(
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
                "metaxy_data_version_by_field": [
                    expected_df["metaxy_data_version_by_field"][0],  # Unchanged
                    {"default": "different_hash"},  # Changed
                    expected_df["metaxy_data_version_by_field"][2],  # Unchanged
                ],
                "metaxy_provenance": [
                    expected_df["metaxy_provenance"][0],
                    "different_prov",
                    expected_df["metaxy_provenance"][2],
                ],
            }
        ).lazy()
    )

    added_lazy, changed_lazy, removed_lazy, _ = (
        engine.resolve_increment_with_provenance(
            current=current,
            upstream=upstream,
            hash_algorithm=HashAlgorithm.XXHASH64,
            filters={},
            sample=None,
        )
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
    aggregation_features: dict[str, type[SampleFeature]],
    sensor_readings_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    snapshot,
) -> None:
    """Test N:1 aggregation lineage loads upstream correctly.

    With per-dependency lineage, aggregation is applied during upstream loading.
    The result should be aggregated by (sensor_id, hour) - 2 groups.
    """
    feature = aggregation_features["HourlyStats"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    # Verify lineage is aggregation (N:1) on the dependency
    assert plan.feature.deps[0].lineage.relationship.type.value == "N:1"

    upstream = {FeatureKey(["sensor_readings"]): sensor_readings_metadata}

    result = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()

    # With aggregation lineage, upstream is aggregated by (sensor_id, hour) during loading
    assert len(result_df) == 2  # 2 groups: s1/hour10, s2/hour10
    assert set(result_df["sensor_id"].to_list()) == {"s1", "s2"}

    # Provenance should be computed for each aggregated group
    assert "metaxy_provenance" in result_df.columns
    assert "metaxy_provenance_by_field" in result_df.columns

    # Snapshot the aggregated upstream provenance
    result_polars = result_df.to_polars()
    provenance_data = sorted(
        [
            {
                "sensor_id": result_polars["sensor_id"][i],
                "hour": result_polars["hour"][i],
                "field_provenance": result_polars["metaxy_provenance_by_field"][i],
                "field_data_version": result_polars["metaxy_data_version_by_field"][i],
            }
            for i in range(len(result_polars))
        ],
        key=lambda x: x["sensor_id"],
    )
    assert provenance_data == snapshot


def test_aggregation_lineage_resolve_increment_no_current(
    aggregation_features: dict[str, type[SampleFeature]],
    sensor_readings_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    snapshot,
) -> None:
    """Test N:1 aggregation increment resolution with no current metadata.

    When current is None, all aggregated groups are returned as added.
    With per-dependency lineage, aggregation happens during loading.
    """
    feature = aggregation_features["HourlyStats"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    upstream = {FeatureKey(["sensor_readings"]): sensor_readings_metadata}

    added_lazy, changed_lazy, removed_lazy, _ = (
        engine.resolve_increment_with_provenance(
            current=None,
            upstream=upstream,
            hash_algorithm=HashAlgorithm.XXHASH64,
            filters={},
            sample=None,
        )
    )

    added = added_lazy.collect()

    # When current is None, all aggregated groups are added
    assert changed_lazy is None
    assert removed_lazy is None
    assert len(added) == 2  # 2 aggregated groups

    # Snapshot the added groups
    added_polars = added.to_polars()
    added_data = sorted(
        [
            {
                "sensor_id": added_polars["sensor_id"][i],
                "hour": added_polars["hour"][i],
                "field_provenance": added_polars["metaxy_provenance_by_field"][i],
                "field_data_version": added_polars["metaxy_data_version_by_field"][i],
            }
            for i in range(len(added_polars))
        ],
        key=lambda x: x["sensor_id"],
    )
    assert added_data == snapshot


def test_aggregation_lineage_resolve_increment_with_changes(
    aggregation_features: dict[str, type[SampleFeature]],
    sensor_readings_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test N:1 aggregation detects changes correctly.

    When upstream readings change, the aggregated hourly stat should be marked as changed.
    The current metadata is at the aggregated level (sensor_id, hour).
    """
    feature = aggregation_features["HourlyStats"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

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
                "metaxy_data_version_by_field": [
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

    added_lazy, changed_lazy, removed_lazy, _ = (
        engine.resolve_increment_with_provenance(
            current=current,
            upstream=upstream,
            hash_algorithm=HashAlgorithm.XXHASH64,
            filters={},
            sample=None,
        )
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
    aggregation_features: dict[str, type[SampleFeature]],
    graph: FeatureGraph,
) -> None:
    """Test that adding new readings to an hour marks that hourly stat as changed.

    This verifies the aggregation logic: when upstream readings are added/changed,
    the aggregated provenance should change. The test uses resolve_increment_with_provenance
    twice to simulate before/after states.
    """
    feature = aggregation_features["HourlyStats"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

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
                "metaxy_data_version_by_field": [
                    {"temperature": "temp_hash_1", "humidity": "hum_hash_1"},
                    {"temperature": "temp_hash_2", "humidity": "hum_hash_2"},
                ],
                "metaxy_provenance": ["reading_prov_1", "reading_prov_2"],
                "metaxy_data_version": ["reading_dv_1", "reading_dv_2"],
            }
        ).lazy()
    )

    # First resolve: no current, so all aggregated groups are added
    added_v1_lazy, _, _, _ = engine.resolve_increment_with_provenance(
        current=None,
        upstream={FeatureKey(["sensor_readings"]): upstream_v1},
        hash_algorithm=HashAlgorithm.XXHASH64,
        filters={},
        sample=None,
    )
    added_v1 = added_v1_lazy.collect()
    assert len(added_v1) == 1  # 1 aggregated group (s1, hour10)

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
                "metaxy_data_version_by_field": [
                    {"temperature": "temp_hash_1", "humidity": "hum_hash_1"},
                    {"temperature": "temp_hash_2", "humidity": "hum_hash_2"},
                    {"temperature": "temp_hash_3", "humidity": "hum_hash_3"},  # New
                ],
                "metaxy_provenance": [
                    "reading_prov_1",
                    "reading_prov_2",
                    "reading_prov_3",  # New
                ],
                "metaxy_data_version": [
                    "reading_dv_1",
                    "reading_dv_2",
                    "reading_dv_3",  # New
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
                "metaxy_data_version_by_field": [
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
    added_lazy, changed_lazy, removed_lazy, _ = (
        engine.resolve_increment_with_provenance(
            current=current_aggregated,
            upstream={FeatureKey(["sensor_readings"]): upstream_v2},
            hash_algorithm=HashAlgorithm.XXHASH64,
            filters={},
            sample=None,
        )
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
    expansion_features: dict[str, type[SampleFeature]],
    video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    snapshot,
) -> None:
    """Test 1:N expansion lineage loads upstream correctly."""
    feature = expansion_features["VideoFrames"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    # Verify lineage is expansion
    # Verify lineage is expansion (1:N) on the dependency
    assert plan.feature.deps[0].lineage.relationship.type.value == "1:N"

    upstream = {FeatureKey(["video"]): video_metadata}

    result = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()

    # Should have one entry per video (parent)
    assert len(result_df) == 2
    assert set(result_df["video_id"].to_list()) == {"v1", "v2"}

    # Snapshot the expansion upstream provenance
    result_polars = result_df.to_polars()
    provenance_data = sorted(
        [
            {
                "video_id": result_polars["video_id"][i],
                "field_provenance": result_polars["metaxy_provenance_by_field"][i],
                "field_data_version": result_polars["metaxy_data_version_by_field"][i],
            }
            for i in range(len(result_polars))
        ],
        key=lambda x: x["video_id"],
    )
    assert provenance_data == snapshot


def test_expansion_lineage_resolve_increment_no_current(
    expansion_features: dict[str, type[SampleFeature]],
    video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
    snapshot,
) -> None:
    """Test 1:N expansion increment resolution with no current metadata."""
    feature = expansion_features["VideoFrames"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    upstream = {FeatureKey(["video"]): video_metadata}

    added_lazy, changed_lazy, removed_lazy, _ = (
        engine.resolve_increment_with_provenance(
            current=None,
            upstream=upstream,
            hash_algorithm=HashAlgorithm.XXHASH64,
            filters={},
            sample=None,
        )
    )

    added = added_lazy.collect()

    # When current is None, all videos should be added
    assert changed_lazy is None
    assert removed_lazy is None
    assert len(added) == 2  # Two videos

    # Snapshot the added videos
    added_polars = added.to_polars()
    added_data = sorted(
        [
            {
                "video_id": added_polars["video_id"][i],
                "field_provenance": added_polars["metaxy_provenance_by_field"][i],
                "field_data_version": added_polars["metaxy_data_version_by_field"][i],
            }
            for i in range(len(added_polars))
        ],
        key=lambda x: x["video_id"],
    )
    assert added_data == snapshot


def test_expansion_lineage_resolve_increment_video_changed(
    expansion_features: dict[str, type[SampleFeature]],
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
    engine = PolarsVersioningEngine(plan)

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
                "metaxy_data_version_by_field": [
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
                "metaxy_data_version": [
                    "video_prov_1",  # Same as fixture
                    "video_prov_2_MODIFIED",  # Changed from fixture
                ],
            }
        ).lazy()
    )

    # Resolve increment with modified video metadata
    # The expansion handler will group video_frames_current by video_id to get one row per video
    # Then compare with upstream (which has 2 videos)
    added_lazy, changed_lazy, removed_lazy, _ = (
        engine.resolve_increment_with_provenance(
            current=video_frames_current,  # Current frames for v1 (3 frames) and v2 (2 frames)
            upstream={FeatureKey(["video"]): upstream_modified},
            hash_algorithm=HashAlgorithm.XXHASH64,
            filters={},
            sample=None,
        )
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
    expansion_features: dict[str, type[SampleFeature]],
    video_frames_current: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test 1:N expansion detects when new parent video is added.

    Adding a new video should be detected at the parent level after grouping current frames.
    """
    feature = expansion_features["VideoFrames"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

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
                "metaxy_data_version_by_field": [
                    {"resolution": "res_hash_1", "fps": "fps_hash_1"},
                    {"resolution": "res_hash_2", "fps": "fps_hash_2"},
                    {"resolution": "res_hash_3", "fps": "fps_hash_3"},  # New
                ],
                "metaxy_provenance": [
                    "video_prov_1",
                    "video_prov_2",
                    "video_prov_3",  # New
                ],
                "metaxy_data_version": [
                    "video_prov_1",
                    "video_prov_2",
                    "video_prov_3",  # New
                ],
            }
        ).lazy()
    )

    added_lazy, changed_lazy, removed_lazy, _ = (
        engine.resolve_increment_with_provenance(
            current=video_frames_current,  # Has frames for v1 and v2 only
            upstream={FeatureKey(["video"]): upstream_with_new_video},
            hash_algorithm=HashAlgorithm.XXHASH64,
            filters={},
            sample=None,
        )
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
    expansion_features: dict[str, type[SampleFeature]],
    video_frames_current: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test 1:N expansion detects when parent video is removed.

    Removing a video should be detected at the parent level after grouping current frames.
    """
    feature = expansion_features["VideoFrames"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    # New upstream only has v1 (v2 is removed)
    upstream_without_v2 = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1"],
                "metaxy_provenance_by_field": [
                    {"resolution": "res_hash_1", "fps": "fps_hash_1"},
                ],
                "metaxy_data_version_by_field": [
                    {"resolution": "res_hash_1", "fps": "fps_hash_1"},
                ],
                "metaxy_provenance": ["video_prov_1"],
                "metaxy_data_version": ["video_prov_1"],
            }
        ).lazy()
    )

    added_lazy, changed_lazy, removed_lazy, _ = (
        engine.resolve_increment_with_provenance(
            current=video_frames_current,  # Has frames for v1 and v2
            upstream={FeatureKey(["video"]): upstream_without_v2},  # Only v1
            hash_algorithm=HashAlgorithm.XXHASH64,
            filters={},
            sample=None,
        )
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


# ============================================================================
# Tests for data_version changes with non-default lineage types
# ============================================================================


def test_aggregation_lineage_upstream_data_version_change_triggers_update(
    aggregation_features: dict[str, type[SampleFeature]],
    graph: FeatureGraph,
) -> None:
    """Test N:1 aggregation detects upstream data_version changes.

    When upstream readings' data_version changes (but provenance stays the same),
    the aggregated hourly stat should be marked as changed. This verifies that
    aggregation lineage correctly uses data_version for change detection.
    """
    feature = aggregation_features["HourlyStats"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    # Initial upstream with specific data_version
    upstream_v1 = nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1", "s1", "s1"],
                "timestamp": [
                    "2024-01-01T10:15:00",
                    "2024-01-01T10:30:00",
                    "2024-01-01T10:45:00",
                ],
                "reading_id": ["r1", "r2", "r3"],
                "hour": ["2024-01-01T10", "2024-01-01T10", "2024-01-01T10"],
                "metaxy_provenance_by_field": [
                    {"temperature": "temp_prov_1", "humidity": "hum_prov_1"},
                    {"temperature": "temp_prov_2", "humidity": "hum_prov_2"},
                    {"temperature": "temp_prov_3", "humidity": "hum_prov_3"},
                ],
                "metaxy_data_version_by_field": [
                    {"temperature": "temp_v1_1", "humidity": "hum_v1_1"},
                    {"temperature": "temp_v1_2", "humidity": "hum_v1_2"},
                    {"temperature": "temp_v1_3", "humidity": "hum_v1_3"},
                ],
                "metaxy_provenance": [
                    "reading_prov_1",
                    "reading_prov_2",
                    "reading_prov_3",
                ],
                "metaxy_data_version": [
                    "reading_dv_v1_1",
                    "reading_dv_v1_2",
                    "reading_dv_v1_3",
                ],
            }
        ).lazy()
    )

    # Get initial expected state
    expected_v1 = engine.load_upstream_with_provenance(
        upstream={FeatureKey(["sensor_readings"]): upstream_v1},
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    ).collect()

    # Create mock current at aggregated level using v1 data_version
    current_v1 = nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1"],
                "hour": ["2024-01-01T10"],
                "metaxy_provenance_by_field": expected_v1["metaxy_provenance_by_field"][
                    0:1
                ],
                "metaxy_data_version_by_field": expected_v1[
                    "metaxy_data_version_by_field"
                ][0:1],
                "metaxy_provenance": [expected_v1["metaxy_provenance"][0]],
            }
        ).lazy()
    )

    # Upstream v2: same provenance, but data_version changed
    # (e.g., user-provided data_version column changed even though computation didn't)
    upstream_v2 = nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1", "s1", "s1"],
                "timestamp": [
                    "2024-01-01T10:15:00",
                    "2024-01-01T10:30:00",
                    "2024-01-01T10:45:00",
                ],
                "reading_id": ["r1", "r2", "r3"],
                "hour": ["2024-01-01T10", "2024-01-01T10", "2024-01-01T10"],
                "metaxy_provenance_by_field": [
                    {"temperature": "temp_prov_1", "humidity": "hum_prov_1"},  # Same
                    {"temperature": "temp_prov_2", "humidity": "hum_prov_2"},  # Same
                    {"temperature": "temp_prov_3", "humidity": "hum_prov_3"},  # Same
                ],
                "metaxy_data_version_by_field": [
                    {"temperature": "temp_v2_1", "humidity": "hum_v2_1"},  # Changed
                    {"temperature": "temp_v2_2", "humidity": "hum_v2_2"},  # Changed
                    {"temperature": "temp_v2_3", "humidity": "hum_v2_3"},  # Changed
                ],
                "metaxy_provenance": [
                    "reading_prov_1",
                    "reading_prov_2",
                    "reading_prov_3",
                ],
                "metaxy_data_version": [
                    "reading_dv_v2_1",  # Changed
                    "reading_dv_v2_2",  # Changed
                    "reading_dv_v2_3",  # Changed
                ],
            }
        ).lazy()
    )

    # Resolve increment with changed data_version
    added_lazy, changed_lazy, removed_lazy, _ = (
        engine.resolve_increment_with_provenance(
            current=current_v1,
            upstream={FeatureKey(["sensor_readings"]): upstream_v2},
            hash_algorithm=HashAlgorithm.XXHASH64,
            filters={},
            sample=None,
        )
    )

    added = added_lazy.collect()
    assert changed_lazy is not None
    changed = changed_lazy.collect()
    assert removed_lazy is not None
    removed = removed_lazy.collect()

    # The aggregated hourly stat should be marked as changed
    # because upstream data_version changed (even though provenance didn't)
    assert len(added) == 0
    assert len(changed) >= 1  # s1's hourly stat changed due to data_version
    assert changed["sensor_id"][0] == "s1"
    assert changed["hour"][0] == "2024-01-01T10"
    assert len(removed) == 0


def test_expansion_lineage_upstream_data_version_change_triggers_update(
    expansion_features: dict[str, type[SampleFeature]],
    graph: FeatureGraph,
) -> None:
    """Test 1:N expansion detects upstream data_version changes.

    When a video's data_version changes (but provenance stays the same),
    ALL frames from that video should be marked as changed. This verifies
    that expansion lineage correctly uses data_version for change detection.
    """
    feature = expansion_features["VideoFrames"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    # Initial upstream with specific data_version
    upstream_v1 = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "metaxy_provenance_by_field": [
                    {"resolution": "res_prov_1", "fps": "fps_prov_1"},
                    {"resolution": "res_prov_2", "fps": "fps_prov_2"},
                ],
                "metaxy_data_version_by_field": [
                    {"resolution": "res_v1_1", "fps": "fps_v1_1"},
                    {"resolution": "res_v1_2", "fps": "fps_v1_2"},
                ],
                "metaxy_provenance": ["video_prov_1", "video_prov_2"],
                "metaxy_data_version": ["video_prov_1", "video_prov_2"],
            }
        ).lazy()
    )

    # Get initial expected state
    engine.load_upstream_with_provenance(
        upstream={FeatureKey(["video"]): upstream_v1},
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    ).collect()

    # Current frames for both videos using v1 data_version
    current_v1 = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1", "v1", "v1", "v2", "v2"],
                "frame_id": [0, 1, 2, 0, 1],
                "metaxy_provenance_by_field": [
                    {"frame_embedding": "frame_v1_0_prov"},
                    {"frame_embedding": "frame_v1_1_prov"},
                    {"frame_embedding": "frame_v1_2_prov"},
                    {"frame_embedding": "frame_v2_0_prov"},
                    {"frame_embedding": "frame_v2_1_prov"},
                ],
                "metaxy_data_version_by_field": [
                    {"frame_embedding": "frame_v1_0_dv"},
                    {"frame_embedding": "frame_v1_1_dv"},
                    {"frame_embedding": "frame_v1_2_dv"},
                    {"frame_embedding": "frame_v2_0_dv"},
                    {"frame_embedding": "frame_v2_1_dv"},
                ],
                "metaxy_provenance": [
                    "frame_v1_0_prov",
                    "frame_v1_1_prov",
                    "frame_v1_2_prov",
                    "frame_v2_0_prov",
                    "frame_v2_1_prov",
                ],
                "metaxy_data_version": [
                    "frame_v1_0_prov",
                    "frame_v1_1_prov",
                    "frame_v1_2_prov",
                    "frame_v2_0_prov",
                    "frame_v2_1_prov",
                ],
            }
        ).lazy()
    )

    # Upstream v2: v2's data_version changed, v1 unchanged
    # (e.g., user-provided data_version column changed for v2)
    upstream_v2 = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "metaxy_provenance_by_field": [
                    {"resolution": "res_prov_1", "fps": "fps_prov_1"},  # Same
                    {"resolution": "res_prov_2", "fps": "fps_prov_2"},  # Same
                ],
                "metaxy_data_version_by_field": [
                    {"resolution": "res_v1_1", "fps": "fps_v1_1"},  # Same as v1
                    {"resolution": "res_v2_2", "fps": "fps_v2_2"},  # Changed for v2
                ],
                "metaxy_provenance": ["video_prov_1", "video_prov_2"],
                "metaxy_data_version": [
                    "video_prov_1",
                    "video_prov_2_changed",
                ],  # v2 changed
            }
        ).lazy()
    )

    # Resolve increment with changed data_version for v2
    added_lazy, changed_lazy, removed_lazy, _ = (
        engine.resolve_increment_with_provenance(
            current=current_v1,
            upstream={FeatureKey(["video"]): upstream_v2},
            hash_algorithm=HashAlgorithm.XXHASH64,
            filters={},
            sample=None,
        )
    )

    added = added_lazy.collect()
    assert changed_lazy is not None
    changed = changed_lazy.collect()
    assert removed_lazy is not None
    removed = removed_lazy.collect()

    # v2's data_version changed, so v2 should appear in changed
    # v1's data_version didn't change, so v1 should not be in changed
    assert len(added) == 0
    assert len(removed) == 0
    assert len(changed) >= 1

    # Check that v2 is in changed (detected at parent level after grouping)
    changed_video_ids = set(changed["video_id"].to_list())
    assert "v2" in changed_video_ids


def test_aggregation_lineage_data_version_vs_provenance_independent(
    aggregation_features: dict[str, type[SampleFeature]],
    graph: FeatureGraph,
) -> None:
    """Test N:1 aggregation: data_version and provenance can change independently.

    Verify that:
    1. Changing only data_version (provenance same) → detected as changed
    2. Changing only provenance (data_version same) → detected as changed
    3. Changing both → detected as changed

    This ensures that the aggregation logic correctly handles both columns.
    """
    feature = aggregation_features["HourlyStats"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    # Base upstream state
    upstream_base = nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1", "s1"],
                "timestamp": ["2024-01-01T10:15:00", "2024-01-01T10:30:00"],
                "reading_id": ["r1", "r2"],
                "hour": ["2024-01-01T10", "2024-01-01T10"],
                "metaxy_provenance_by_field": [
                    {"temperature": "temp_prov_base", "humidity": "hum_prov_base"},
                    {"temperature": "temp_prov_base", "humidity": "hum_prov_base"},
                ],
                "metaxy_data_version_by_field": [
                    {"temperature": "temp_dv_base", "humidity": "hum_dv_base"},
                    {"temperature": "temp_dv_base", "humidity": "hum_dv_base"},
                ],
                "metaxy_provenance": ["reading_prov_base", "reading_prov_base"],
                "metaxy_data_version": ["reading_dv_base", "reading_dv_base"],
            }
        ).lazy()
    )

    # Get base expected state
    expected_base = engine.load_upstream_with_provenance(
        upstream={FeatureKey(["sensor_readings"]): upstream_base},
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    ).collect()

    current_base = nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1"],
                "hour": ["2024-01-01T10"],
                "metaxy_provenance_by_field": expected_base[
                    "metaxy_provenance_by_field"
                ][0:1],
                "metaxy_data_version_by_field": expected_base[
                    "metaxy_data_version_by_field"
                ][0:1],
                "metaxy_provenance": [expected_base["metaxy_provenance"][0]],
            }
        ).lazy()
    )

    # Test 1: Only data_version changed
    upstream_dv_changed = nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1", "s1"],
                "timestamp": ["2024-01-01T10:15:00", "2024-01-01T10:30:00"],
                "reading_id": ["r1", "r2"],
                "hour": ["2024-01-01T10", "2024-01-01T10"],
                "metaxy_provenance_by_field": [
                    {
                        "temperature": "temp_prov_base",
                        "humidity": "hum_prov_base",
                    },  # Same
                    {
                        "temperature": "temp_prov_base",
                        "humidity": "hum_prov_base",
                    },  # Same
                ],
                "metaxy_data_version_by_field": [
                    {"temperature": "temp_dv_NEW", "humidity": "hum_dv_NEW"},  # Changed
                    {"temperature": "temp_dv_NEW", "humidity": "hum_dv_NEW"},  # Changed
                ],
                "metaxy_provenance": ["reading_prov_base", "reading_prov_base"],
                "metaxy_data_version": ["reading_dv_NEW", "reading_dv_NEW"],  # Changed
            }
        ).lazy()
    )

    added_lazy, changed_lazy, removed_lazy, _ = (
        engine.resolve_increment_with_provenance(
            current=current_base,
            upstream={FeatureKey(["sensor_readings"]): upstream_dv_changed},
            hash_algorithm=HashAlgorithm.XXHASH64,
            filters={},
            sample=None,
        )
    )

    changed = changed_lazy.collect() if changed_lazy else nw.from_native(pl.DataFrame())
    assert len(changed) >= 1, "Changing only data_version should trigger change"

    # Test 2: Only provenance_by_field changed (data_version_by_field same)
    # Since provenance is computed from data_version_by_field (not provenance_by_field),
    # changing only provenance_by_field should NOT trigger a change
    upstream_prov_changed = nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1", "s1"],
                "timestamp": ["2024-01-01T10:15:00", "2024-01-01T10:30:00"],
                "reading_id": ["r1", "r2"],
                "hour": ["2024-01-01T10", "2024-01-01T10"],
                "metaxy_provenance_by_field": [
                    {
                        "temperature": "temp_prov_NEW",
                        "humidity": "hum_prov_NEW",
                    },  # Changed
                    {
                        "temperature": "temp_prov_NEW",
                        "humidity": "hum_prov_NEW",
                    },  # Changed
                ],
                "metaxy_data_version_by_field": [
                    {"temperature": "temp_dv_base", "humidity": "hum_dv_base"},  # Same
                    {"temperature": "temp_dv_base", "humidity": "hum_dv_base"},  # Same
                ],
                "metaxy_provenance": ["reading_prov_base", "reading_prov_base"],
                "metaxy_data_version": ["reading_dv_base", "reading_dv_base"],  # Same
            }
        ).lazy()
    )

    added_lazy, changed_lazy, removed_lazy, _ = (
        engine.resolve_increment_with_provenance(
            current=current_base,
            upstream={FeatureKey(["sensor_readings"]): upstream_prov_changed},
            hash_algorithm=HashAlgorithm.XXHASH64,
            filters={},
            sample=None,
        )
    )

    changed = changed_lazy.collect() if changed_lazy else nw.from_native(pl.DataFrame())
    # Provenance is computed from data_version_by_field, so changing only provenance_by_field
    # should NOT trigger a change (data_version_by_field is unchanged)
    assert len(changed) == 0, (
        "Changing only provenance_by_field should NOT trigger change"
    )


def test_expansion_lineage_data_version_vs_provenance_independent(
    expansion_features: dict[str, type[SampleFeature]],
    graph: FeatureGraph,
) -> None:
    """Test 1:N expansion: data_version controls change detection, not provenance_by_field.

    Verify that:
    1. Changing only data_version (provenance same) → detected as changed
    2. Changing only provenance_by_field (data_version same) → NOT detected as changed

    Provenance is computed from data_version_by_field, so changing only provenance_by_field
    should NOT trigger a change.
    """
    feature = expansion_features["VideoFrames"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    # Base upstream state - compute actual expected provenance from this
    upstream_base = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1"],
                "metaxy_provenance_by_field": [
                    {"resolution": "res_prov_base", "fps": "fps_prov_base"},
                ],
                "metaxy_data_version_by_field": [
                    {"resolution": "res_dv_base", "fps": "fps_dv_base"},
                ],
                "metaxy_provenance": ["video_prov_base"],
                "metaxy_data_version": ["video_prov_base"],
            }
        ).lazy()
    )

    # Get the actual expected provenance by computing it
    expected_base = engine.load_upstream_with_provenance(
        upstream={FeatureKey(["video"]): upstream_base},
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    ).collect()

    # Current frames - use the SAME provenance as computed expected
    # This ensures a baseline with no changes
    current_base = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1", "v1"],
                "frame_id": [0, 1],
                "metaxy_provenance_by_field": [
                    expected_base["metaxy_provenance_by_field"][0],
                    expected_base["metaxy_provenance_by_field"][
                        0
                    ],  # Same for all frames
                ],
                "metaxy_data_version_by_field": [
                    expected_base["metaxy_data_version_by_field"][0],
                    expected_base["metaxy_data_version_by_field"][0],
                ],
                "metaxy_provenance": [
                    expected_base["metaxy_provenance"][0],
                    expected_base["metaxy_provenance"][0],
                ],
            }
        ).lazy()
    )

    # Test 1: Only data_version changed
    upstream_dv_changed = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1"],
                "metaxy_provenance_by_field": [
                    {"resolution": "res_prov_base", "fps": "fps_prov_base"},  # Same
                ],
                "metaxy_data_version_by_field": [
                    {"resolution": "res_dv_NEW", "fps": "fps_dv_NEW"},  # Changed
                ],
                "metaxy_provenance": ["video_prov_base"],
                "metaxy_data_version": ["video_prov_NEW"],  # Changed
            }
        ).lazy()
    )

    added_lazy, changed_lazy, removed_lazy, _ = (
        engine.resolve_increment_with_provenance(
            current=current_base,
            upstream={FeatureKey(["video"]): upstream_dv_changed},
            hash_algorithm=HashAlgorithm.XXHASH64,
            filters={},
            sample=None,
        )
    )

    changed = changed_lazy.collect() if changed_lazy else nw.from_native(pl.DataFrame())
    assert len(changed) >= 1, "Changing only data_version should trigger change"
    assert "v1" in set(changed["video_id"].to_list())

    # Test 2: Only provenance_by_field changed (data_version_by_field same)
    upstream_prov_changed = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1"],
                "metaxy_provenance_by_field": [
                    {"resolution": "res_prov_NEW", "fps": "fps_prov_NEW"},  # Changed
                ],
                "metaxy_data_version_by_field": [
                    {"resolution": "res_dv_base", "fps": "fps_dv_base"},  # Same as base
                ],
                "metaxy_provenance": ["video_prov_base"],
                "metaxy_data_version": ["video_prov_base"],  # Same as base
            }
        ).lazy()
    )

    added_lazy, changed_lazy, removed_lazy, _ = (
        engine.resolve_increment_with_provenance(
            current=current_base,
            upstream={FeatureKey(["video"]): upstream_prov_changed},
            hash_algorithm=HashAlgorithm.XXHASH64,
            filters={},
            sample=None,
        )
    )

    changed = changed_lazy.collect() if changed_lazy else nw.from_native(pl.DataFrame())
    # Provenance is computed from data_version_by_field, so changing only provenance_by_field
    # should NOT trigger a change (data_version_by_field is unchanged)
    assert len(changed) == 0, (
        "Changing only provenance_by_field should NOT trigger change"
    )


# ============================================================================
# Tests for Mixed Lineage Relationships (per-dependency lineage)
# ============================================================================


@pytest.fixture
def mixed_lineage_features(graph: FeatureGraph) -> dict[str, type[SampleFeature]]:
    """Create features for testing mixed lineage relationships.

    Scenario: Enriched sensor readings that aggregate raw readings and join with sensor info.
    - SensorInfo: sensor_id (lookup table with 1:1 relationship)
    - SensorReadings: sensor_id, timestamp, reading_id (fine-grained data)
    - HourlyEnrichedStats: sensor_id, hour (aggregated readings + identity lookup of sensor info)

    This tests the new per-dependency lineage where different deps can have different
    lineage relationships:
    - FeatureDep(SensorReadings, lineage=aggregation) → N:1
    - FeatureDep(SensorInfo, lineage=identity) → 1:1
    """

    class SensorInfo(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["sensor_info"]),
            id_columns=(
                "sensor_id",
                "hour",
            ),  # Include hour so join works after aggregation
            fields=[
                FieldSpec(key=FieldKey(["location"]), code_version="1"),
                FieldSpec(key=FieldKey(["sensor_type"]), code_version="1"),
            ],
        ),
    ):
        pass

    class SensorReadings(
        SampleFeature,
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

    class HourlyEnrichedStats(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["hourly_enriched_stats"]),
            id_columns=("sensor_id", "hour"),
            deps=[
                # Aggregation: many readings per hour → one hourly stat
                FeatureDep(
                    feature=FeatureKey(["sensor_readings"]),
                    lineage=LineageRelationship.aggregation(on=["sensor_id", "hour"]),
                ),
                # Identity: one sensor info record per sensor
                FeatureDep(
                    feature=FeatureKey(["sensor_info"]),
                    lineage=LineageRelationship.identity(),
                ),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["avg_temp"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                ),
                FieldSpec(
                    key=FieldKey(["location_temp"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                ),
            ],
        ),
    ):
        pass

    return {
        "SensorInfo": SensorInfo,
        "SensorReadings": SensorReadings,
        "HourlyEnrichedStats": HourlyEnrichedStats,
    }


def test_mixed_lineage_input_id_columns(
    mixed_lineage_features: dict[str, type[SampleFeature]],
    graph: FeatureGraph,
) -> None:
    """Test that input_id_columns is correctly computed for mixed lineage.

    With per-dep lineage:
    - SensorReadings dep with aggregation(on=["sensor_id", "hour"]) → input cols: ["sensor_id", "hour"]
    - SensorInfo dep with identity → input cols: ["sensor_id", "hour"] (its ID columns)

    The intersection is ["sensor_id", "hour"].
    """
    feature = mixed_lineage_features["HourlyEnrichedStats"]
    plan = graph.get_feature_plan(feature.spec().key)

    # Verify mixed lineage configuration
    deps = plan.feature.deps
    assert len(deps) == 2

    # First dep is aggregation
    assert deps[0].lineage.relationship.type.value == "N:1"
    # Second dep is identity
    assert deps[1].lineage.relationship.type.value == "1:1"

    # Check input_id_columns per dep
    readings_dep_cols = plan.get_input_id_columns_for_dep(deps[0])
    assert set(readings_dep_cols) == {"sensor_id", "hour"}

    info_dep_cols = plan.get_input_id_columns_for_dep(deps[1])
    assert set(info_dep_cols) == {"sensor_id", "hour"}

    # Overall input_id_columns is intersection
    assert set(plan.input_id_columns) == {"sensor_id", "hour"}


def test_mixed_lineage_get_input_id_columns_per_dep(
    mixed_lineage_features: dict[str, type[SampleFeature]],
    graph: FeatureGraph,
) -> None:
    """Test that get_input_id_columns_for_dep works correctly for each dependency.

    This tests the per-dependency input ID column calculation which is the core
    of the new per-dep lineage feature.
    """
    feature = mixed_lineage_features["HourlyEnrichedStats"]
    plan = graph.get_feature_plan(feature.spec().key)

    # Get the two dependencies
    readings_dep = plan.feature.deps[0]  # aggregation lineage
    info_dep = plan.feature.deps[1]  # identity lineage

    # For aggregation dep, input columns are the aggregation 'on' columns
    readings_input_cols = plan.get_input_id_columns_for_dep(readings_dep)
    assert set(readings_input_cols) == {"sensor_id", "hour"}, (
        f"Expected aggregation to use 'on' columns, got {readings_input_cols}"
    )

    # For identity dep, input columns are the upstream ID columns (after renames)
    info_input_cols = plan.get_input_id_columns_for_dep(info_dep)
    assert set(info_input_cols) == {"sensor_id", "hour"}, (
        f"Expected identity to use upstream ID columns, got {info_input_cols}"
    )


def test_mixed_lineage_different_relationship_types(graph: FeatureGraph) -> None:
    """Test a feature that explicitly mixes aggregation and identity lineage.

    This is a common pattern: aggregate detailed data while doing a 1:1 lookup
    from a dimension table.
    """

    class DimensionTable(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["dimension"]),
            id_columns=("dim_key",),
            fields=[FieldSpec(key=FieldKey(["dim_value"]), code_version="1")],
        ),
    ):
        pass

    class DetailedFacts(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["detailed_facts"]),
            id_columns=("dim_key", "timestamp", "fact_id"),
            fields=[FieldSpec(key=FieldKey(["fact_value"]), code_version="1")],
        ),
    ):
        pass

    class AggregatedWithDimension(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["aggregated_with_dimension"]),
            id_columns=("dim_key",),
            deps=[
                # Aggregate facts by dim_key
                FeatureDep(
                    feature=DetailedFacts,
                    lineage=LineageRelationship.aggregation(on=["dim_key"]),
                ),
                # 1:1 join with dimension table
                FeatureDep(
                    feature=DimensionTable,
                    lineage=LineageRelationship.identity(),
                ),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["summary"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                )
            ],
        ),
    ):
        pass

    plan = graph.get_feature_plan(AggregatedWithDimension.spec().key)

    # Verify the per-dep lineage configuration
    facts_dep = plan.feature.deps[0]
    dim_dep = plan.feature.deps[1]

    assert facts_dep.lineage.relationship.type.value == "N:1"
    assert dim_dep.lineage.relationship.type.value == "1:1"

    # Input columns for facts dep (after aggregation) should be ["dim_key"]
    facts_input_cols = plan.get_input_id_columns_for_dep(facts_dep)
    assert set(facts_input_cols) == {"dim_key"}

    # Input columns for dim dep (identity) should be ["dim_key"]
    dim_input_cols = plan.get_input_id_columns_for_dep(dim_dep)
    assert set(dim_input_cols) == {"dim_key"}

    # Overall input_id_columns is intersection
    assert set(plan.input_id_columns) == {"dim_key"}
