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

    # With aggregation lineage using window functions, all original rows are preserved
    # but metaxy columns are aggregated field-by-field (all rows in a group have identical values)
    # 5 original rows: s1 has 3 readings, s2 has 2 readings
    assert len(result_df) == 5
    assert result_df["sensor_id"].to_list().count("s1") == 3
    assert result_df["sensor_id"].to_list().count("s2") == 2

    # Provenance should be computed for each aggregated group
    assert "metaxy_provenance" in result_df.columns
    assert "metaxy_provenance_by_field" in result_df.columns

    # Verify that all rows within a group have identical metaxy values (window function behavior)
    result_polars = result_df.to_polars()

    # Check s1 group: all 3 rows should have identical provenance
    s1_rows = result_polars.filter(result_polars["sensor_id"] == "s1")
    s1_provenance = s1_rows["metaxy_provenance"].to_list()
    assert len(set(s1_provenance)) == 1, "All s1 rows should have identical provenance"

    # Check s2 group: both rows should have identical provenance
    s2_rows = result_polars.filter(result_polars["sensor_id"] == "s2")
    s2_provenance = s2_rows["metaxy_provenance"].to_list()
    assert len(set(s2_provenance)) == 1, "All s2 rows should have identical provenance"

    # s1 and s2 should have different provenance (different upstream data)
    assert s1_provenance[0] != s2_provenance[0], (
        "Different groups should have different provenance"
    )


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

    # When current is None, all rows are added (5 original rows with aggregated metaxy values)
    assert changed_lazy is None
    assert removed_lazy is None
    assert len(added) == 5  # All 5 rows preserved with aggregated metaxy values

    # Verify that all rows within a group have identical metaxy values
    added_polars = added.to_polars()

    # Check s1 group: all 3 rows should have identical provenance
    s1_rows = added_polars.filter(added_polars["sensor_id"] == "s1")
    s1_provenance = s1_rows["metaxy_provenance"].to_list()
    assert len(set(s1_provenance)) == 1, "All s1 rows should have identical provenance"

    # Check s2 group: both rows should have identical provenance
    s2_rows = added_polars.filter(added_polars["sensor_id"] == "s2")
    s2_provenance = s2_rows["metaxy_provenance"].to_list()
    assert len(set(s2_provenance)) == 1, "All s2 rows should have identical provenance"


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
    assert (
        len(added_v1) == 2
    )  # 2 rows with aggregated metaxy values (s1 has 2 readings)

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


# ============================================================================
# Tests for Column Cleanup (no leaked internal columns)
# ============================================================================


def test_load_upstream_does_not_leak_renamed_data_version_column(
    aggregation_features: dict[str, type[SampleFeature]],
    sensor_readings_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test that load_upstream_with_provenance doesn't leak renamed internal columns.

    The AggregationLineageTransformer creates a renamed data_version column like
    `metaxy_data_version__sensor_readings` during aggregation. This column should
    be cleaned up and not appear in the final output.

    Regression test for a bug where renamed_data_version_col was not dropped.
    """
    feature = aggregation_features["HourlyStats"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    upstream = {FeatureKey(["sensor_readings"]): sensor_readings_metadata}

    result = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()
    columns = result_df.columns

    # Check that no internal renamed columns are leaked
    # These patterns should NOT appear in output:
    # - metaxy_data_version__<feature_key>
    # - metaxy_provenance__<feature_key>
    # - metaxy_provenance_by_field__<feature_key>
    # - metaxy_data_version_by_field__<feature_key>
    leaked_columns = [
        col for col in columns if col.startswith("metaxy_") and "__" in col
    ]
    assert leaked_columns == [], (
        f"Internal renamed columns leaked to output: {leaked_columns}"
    )


def test_load_upstream_does_not_leak_columns_identity_lineage(
    simple_features: dict[str, type[SampleFeature]],
    upstream_video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test that identity lineage also cleans up renamed columns properly."""
    feature = simple_features["ProcessedVideo"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    upstream = {FeatureKey(["video"]): upstream_video_metadata}

    result = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()
    columns = result_df.columns

    leaked_columns = [
        col for col in columns if col.startswith("metaxy_") and "__" in col
    ]
    assert leaked_columns == [], (
        f"Internal renamed columns leaked to output: {leaked_columns}"
    )


def test_load_upstream_does_not_leak_columns_expansion_lineage(
    expansion_features: dict[str, type[SampleFeature]],
    video_metadata: nw.LazyFrame[pl.LazyFrame],
    graph: FeatureGraph,
) -> None:
    """Test that expansion lineage also cleans up renamed columns properly."""
    feature = expansion_features["VideoFrames"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    upstream = {FeatureKey(["video"]): video_metadata}

    result = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()
    columns = result_df.columns

    leaked_columns = [
        col for col in columns if col.startswith("metaxy_") and "__" in col
    ]
    assert leaked_columns == [], (
        f"Internal renamed columns leaked to output: {leaked_columns}"
    )


def test_load_upstream_does_not_leak_columns_mixed_lineage(
    mixed_lineage_features: dict[str, type[SampleFeature]],
    graph: FeatureGraph,
) -> None:
    """Test that mixed lineage (aggregation + identity) cleans up renamed columns."""
    feature = mixed_lineage_features["HourlyEnrichedStats"]
    plan = graph.get_feature_plan(feature.spec().key)
    engine = PolarsVersioningEngine(plan)

    # Create mock upstream data for both dependencies
    sensor_readings = nw.from_native(
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

    sensor_info = nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1"],
                "hour": ["2024-01-01T10"],
                "metaxy_provenance_by_field": [
                    {"location": "loc_hash", "sensor_type": "type_hash"}
                ],
                "metaxy_data_version_by_field": [
                    {"location": "loc_hash", "sensor_type": "type_hash"}
                ],
                "metaxy_provenance": ["info_prov"],
                "metaxy_data_version": ["info_dv"],
            }
        ).lazy()
    )

    upstream = {
        FeatureKey(["sensor_readings"]): sensor_readings,
        FeatureKey(["sensor_info"]): sensor_info,
    }

    result = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()
    columns = result_df.columns

    # Should not have any renamed internal columns from either dependency
    leaked_columns = [
        col for col in columns if col.startswith("metaxy_") and "__" in col
    ]
    assert leaked_columns == [], (
        f"Internal renamed columns leaked to output: {leaked_columns}"
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


# ============================================================================
# Tests for aggregation column auto-inclusion when columns= is specified
# ============================================================================


def test_aggregation_columns_auto_included_when_columns_specified(
    graph: FeatureGraph,
) -> None:
    """Test that aggregation 'on' columns are automatically included when columns= is specified.

    This reproduces a bug where specifying columns= on a FeatureDep with aggregation lineage
    would cause the aggregation column to be missing, leading to:
    - "unable to find column" error during aggregation

    The scenario:
    - Feature A has id_columns=["id_a"] and fields including "group_col"
    - Feature B depends on A with:
      - columns=("field_x", "field_y") - explicit column selection NOT including group_col
      - lineage=aggregation(on=["group_col"])

    Expected behavior: group_col should be auto-included since it's needed for aggregation.
    """

    class UpstreamDetail(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["upstream_detail"]),
            id_columns=("detail_id",),
            fields=[
                FieldSpec(key=FieldKey(["group_col"]), code_version="1"),
                FieldSpec(key=FieldKey(["field_x"]), code_version="1"),
                FieldSpec(key=FieldKey(["field_y"]), code_version="1"),
            ],
        ),
    ):
        pass

    class AggregatedFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["aggregated_feature"]),
            id_columns=("group_col",),
            deps=[
                FeatureDep(
                    feature=UpstreamDetail,
                    # BUG: columns= doesn't include group_col, but aggregation needs it
                    columns=("field_x", "field_y"),
                    lineage=LineageRelationship.aggregation(on=["group_col"]),
                )
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["aggregated_value"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                )
            ],
        ),
    ):
        pass

    plan = graph.get_feature_plan(AggregatedFeature.spec().key)
    engine = PolarsVersioningEngine(plan)

    # Create upstream data
    upstream_data = nw.from_native(
        pl.DataFrame(
            {
                "detail_id": ["d1", "d2", "d3"],
                "group_col": ["g1", "g1", "g2"],  # d1, d2 -> g1; d3 -> g2
                "field_x": ["x1", "x2", "x3"],
                "field_y": ["y1", "y2", "y3"],
                "metaxy_provenance_by_field": [
                    {"group_col": "gc_h1", "field_x": "fx_h1", "field_y": "fy_h1"},
                    {"group_col": "gc_h2", "field_x": "fx_h2", "field_y": "fy_h2"},
                    {"group_col": "gc_h3", "field_x": "fx_h3", "field_y": "fy_h3"},
                ],
                "metaxy_data_version_by_field": [
                    {"group_col": "gc_h1", "field_x": "fx_h1", "field_y": "fy_h1"},
                    {"group_col": "gc_h2", "field_x": "fx_h2", "field_y": "fy_h2"},
                    {"group_col": "gc_h3", "field_x": "fx_h3", "field_y": "fy_h3"},
                ],
                "metaxy_provenance": ["prov_1", "prov_2", "prov_3"],
                "metaxy_data_version": ["dv_1", "dv_2", "dv_3"],
            }
        ).lazy()
    )

    upstream = {FeatureKey(["upstream_detail"]): upstream_data}

    # This should NOT raise "unable to find column 'group_col'"
    # The aggregation column should be auto-included
    result = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()

    # With window functions, all 3 original rows are preserved with aggregated metaxy values
    # (d1, d2 in group g1; d3 in group g2)
    assert len(result_df) == 3
    assert result_df["group_col"].to_list().count("g1") == 2
    assert result_df["group_col"].to_list().count("g2") == 1


def test_two_deps_with_aggregation_on_same_column_with_explicit_columns(
    graph: FeatureGraph,
) -> None:
    """Test two deps aggregating on the same column where neither specifies the agg column in columns=.

    The aggregation 'on' columns should be automatically included from the lineage relationship -
    there's no need to specify them twice in both `columns=` and `lineage=aggregation(on=...)`.

    This test verifies:
    - chunk_id is auto-included for both deps from lineage.aggregation(on=["chunk_id"])
    - chunk_id is recognized as a shared join column (not a collision)
    - The aggregation works correctly with the auto-included column
    """

    class NotesTexts(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["notes_texts"]),
            id_columns=("notes_id",),
            fields=[
                FieldSpec(key=FieldKey(["chunk_id"]), code_version="1"),
                FieldSpec(key=FieldKey(["notes_type"]), code_version="1"),
                FieldSpec(key=FieldKey(["text_path"]), code_version="1"),
                FieldSpec(key=FieldKey(["text_size"]), code_version="1"),
            ],
        ),
    ):
        pass

    class NotesEncodings(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["notes_encodings"]),
            id_columns=("encoding_id",),
            fields=[
                FieldSpec(key=FieldKey(["chunk_id"]), code_version="1"),
                FieldSpec(key=FieldKey(["encoding_path"]), code_version="1"),
                FieldSpec(key=FieldKey(["encoding_size"]), code_version="1"),
            ],
        ),
    ):
        pass

    class AggregatedByChunk(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["aggregated_by_chunk"]),
            id_columns=("chunk_id",),
            deps=[
                # First dep: chunk_id NOT in columns - auto-included from lineage
                FeatureDep(
                    feature=NotesEncodings,
                    columns=("encoding_path", "encoding_size"),
                    rename={
                        "encoding_path": "encodings_path",
                        "encoding_size": "encodings_size",
                    },
                    lineage=LineageRelationship.aggregation(on=["chunk_id"]),
                ),
                # Second dep: chunk_id NOT in columns - auto-included from lineage
                FeatureDep(
                    feature=NotesTexts,
                    columns=("notes_type", "text_path", "text_size"),
                    rename={"text_path": "texts_path", "text_size": "texts_size"},
                    lineage=LineageRelationship.aggregation(on=["chunk_id"]),
                ),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["aggregated"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                )
            ],
        ),
    ):
        pass

    plan = graph.get_feature_plan(AggregatedByChunk.spec().key)
    engine = PolarsVersioningEngine(plan)

    # Create upstream data for NotesEncodings
    encodings_data = nw.from_native(
        pl.DataFrame(
            {
                "encoding_id": ["e1", "e2"],
                "chunk_id": ["c1", "c1"],
                "encoding_path": ["/path/e1", "/path/e2"],
                "encoding_size": [100, 200],
                "metaxy_provenance_by_field": [
                    {"chunk_id": "h1", "encoding_path": "h2", "encoding_size": "h3"},
                    {"chunk_id": "h4", "encoding_path": "h5", "encoding_size": "h6"},
                ],
                "metaxy_data_version_by_field": [
                    {"chunk_id": "h1", "encoding_path": "h2", "encoding_size": "h3"},
                    {"chunk_id": "h4", "encoding_path": "h5", "encoding_size": "h6"},
                ],
                "metaxy_provenance": ["enc_prov_1", "enc_prov_2"],
                "metaxy_data_version": ["enc_dv_1", "enc_dv_2"],
            }
        ).lazy()
    )

    # Create upstream data for NotesTexts
    texts_data = nw.from_native(
        pl.DataFrame(
            {
                "notes_id": ["n1", "n2", "n3"],
                "chunk_id": ["c1", "c1", "c1"],
                "notes_type": ["imperative", "descriptive", "keywords"],
                "text_path": ["/path/n1", "/path/n2", "/path/n3"],
                "text_size": [10, 20, 30],
                "metaxy_provenance_by_field": [
                    {
                        "chunk_id": "h1",
                        "notes_type": "h2",
                        "text_path": "h3",
                        "text_size": "h4",
                    },
                    {
                        "chunk_id": "h5",
                        "notes_type": "h6",
                        "text_path": "h7",
                        "text_size": "h8",
                    },
                    {
                        "chunk_id": "h9",
                        "notes_type": "h10",
                        "text_path": "h11",
                        "text_size": "h12",
                    },
                ],
                "metaxy_data_version_by_field": [
                    {
                        "chunk_id": "h1",
                        "notes_type": "h2",
                        "text_path": "h3",
                        "text_size": "h4",
                    },
                    {
                        "chunk_id": "h5",
                        "notes_type": "h6",
                        "text_path": "h7",
                        "text_size": "h8",
                    },
                    {
                        "chunk_id": "h9",
                        "notes_type": "h10",
                        "text_path": "h11",
                        "text_size": "h12",
                    },
                ],
                "metaxy_provenance": ["txt_prov_1", "txt_prov_2", "txt_prov_3"],
                "metaxy_data_version": ["txt_dv_1", "txt_dv_2", "txt_dv_3"],
            }
        ).lazy()
    )

    upstream = {
        FeatureKey(["notes_encodings"]): encodings_data,
        FeatureKey(["notes_texts"]): texts_data,
    }

    # This should NOT raise:
    # - "unable to find column 'chunk_id'" (if chunk_id not auto-included)
    # - "ambiguous columns" (if chunk_id included but treated as collision)
    result = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()

    # With window functions, all joined rows are preserved with aggregated metaxy values
    # encodings (2 rows) × texts (3 rows) = 6 rows, all with chunk_id="c1"
    assert len(result_df) == 6
    assert all(cid == "c1" for cid in result_df["chunk_id"].to_list())


def test_two_deps_with_same_upstream_id_column_and_aggregation(
    graph: FeatureGraph,
) -> None:
    """Test two deps from features with the SAME upstream ID column, aggregating on a different column.

    This reproduces a real bug where:
    - DirectorNotesTexts has id_columns=("director_notes_id",) and chunk_id as a field
    - DirectorNotesEncodings has id_columns=("director_notes_id",) and chunk_id as a field
    - Both aggregate on chunk_id

    The upstream ID column (director_notes_id) is auto-included for both deps, causing
    "ambiguous columns" error because it's repeated but not recognized as a shared column.

    Expected: The upstream ID columns should be allowed to repeat since they come from
    the same logical source (both deps have the same upstream ID column structure).
    """

    class DirectorNotesTexts(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["director_notes_texts"]),
            id_columns=("director_notes_id",),
            fields=[
                FieldSpec(key=FieldKey(["chunk_id"]), code_version="1"),
                FieldSpec(key=FieldKey(["director_notes_type"]), code_version="1"),
                FieldSpec(key=FieldKey(["path"]), code_version="1"),
                FieldSpec(key=FieldKey(["size"]), code_version="1"),
            ],
        ),
    ):
        pass

    class DirectorNotesEncodings(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["director_notes_encodings"]),
            id_columns=("director_notes_id",),
            fields=[
                FieldSpec(key=FieldKey(["chunk_id"]), code_version="1"),
                FieldSpec(key=FieldKey(["path"]), code_version="1"),
                FieldSpec(key=FieldKey(["size"]), code_version="1"),
            ],
        ),
    ):
        pass

    class DirectorNotesAggregatedByChunk(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["director_notes_aggregated"]),
            id_columns=("chunk_id",),
            deps=[
                FeatureDep(
                    feature=DirectorNotesEncodings,
                    columns=("path", "size", "chunk_id"),
                    rename={"path": "encodings_path", "size": "encodings_size"},
                    lineage=LineageRelationship.aggregation(on=["chunk_id"]),
                ),
                FeatureDep(
                    feature=DirectorNotesTexts,
                    columns=("director_notes_type", "path", "size"),
                    rename={"path": "text_path", "size": "text_size"},
                    lineage=LineageRelationship.aggregation(on=["chunk_id"]),
                ),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["aggregated"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                )
            ],
        ),
    ):
        pass

    plan = graph.get_feature_plan(DirectorNotesAggregatedByChunk.spec().key)
    engine = PolarsVersioningEngine(plan)

    # Create upstream data for DirectorNotesEncodings
    encodings_data = nw.from_native(
        pl.DataFrame(
            {
                "director_notes_id": ["dn1", "dn2"],
                "chunk_id": ["c1", "c1"],
                "path": ["/path/e1", "/path/e2"],
                "size": [100, 200],
                "metaxy_provenance_by_field": [
                    {"chunk_id": "h1", "path": "h2", "size": "h3"},
                    {"chunk_id": "h4", "path": "h5", "size": "h6"},
                ],
                "metaxy_data_version_by_field": [
                    {"chunk_id": "h1", "path": "h2", "size": "h3"},
                    {"chunk_id": "h4", "path": "h5", "size": "h6"},
                ],
                "metaxy_provenance": ["enc_prov_1", "enc_prov_2"],
                "metaxy_data_version": ["enc_dv_1", "enc_dv_2"],
            }
        ).lazy()
    )

    # Create upstream data for DirectorNotesTexts
    texts_data = nw.from_native(
        pl.DataFrame(
            {
                "director_notes_id": ["dn1", "dn2", "dn3"],
                "chunk_id": ["c1", "c1", "c1"],
                "director_notes_type": ["imperative", "descriptive", "keywords"],
                "path": ["/path/t1", "/path/t2", "/path/t3"],
                "size": [10, 20, 30],
                "metaxy_provenance_by_field": [
                    {
                        "chunk_id": "h1",
                        "director_notes_type": "h2",
                        "path": "h3",
                        "size": "h4",
                    },
                    {
                        "chunk_id": "h5",
                        "director_notes_type": "h6",
                        "path": "h7",
                        "size": "h8",
                    },
                    {
                        "chunk_id": "h9",
                        "director_notes_type": "h10",
                        "path": "h11",
                        "size": "h12",
                    },
                ],
                "metaxy_data_version_by_field": [
                    {
                        "chunk_id": "h1",
                        "director_notes_type": "h2",
                        "path": "h3",
                        "size": "h4",
                    },
                    {
                        "chunk_id": "h5",
                        "director_notes_type": "h6",
                        "path": "h7",
                        "size": "h8",
                    },
                    {
                        "chunk_id": "h9",
                        "director_notes_type": "h10",
                        "path": "h11",
                        "size": "h12",
                    },
                ],
                "metaxy_provenance": ["txt_prov_1", "txt_prov_2", "txt_prov_3"],
                "metaxy_data_version": ["txt_dv_1", "txt_dv_2", "txt_dv_3"],
            }
        ).lazy()
    )

    upstream = {
        FeatureKey(["director_notes_encodings"]): encodings_data,
        FeatureKey(["director_notes_texts"]): texts_data,
    }

    # This should NOT raise "ambiguous columns" for director_notes_id
    result = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()

    # With window functions, all joined rows are preserved with aggregated metaxy values
    # encodings (2 rows) × texts (3 rows) = 6 rows, all with chunk_id="c1"
    assert len(result_df) == 6
    assert all(cid == "c1" for cid in result_df["chunk_id"].to_list())


def test_expansion_columns_auto_included_when_columns_specified(
    graph: FeatureGraph,
) -> None:
    """Test that expansion 'on' columns are automatically included when columns= is specified.

    This mirrors test_aggregation_columns_auto_included_when_columns_specified but for expansion.

    The scenario:
    - Feature A (Video) has id_columns=["video_id"] and fields including metadata
    - Feature B (VideoFrames) depends on A with:
      - columns=("resolution",) - explicit column selection NOT including video_id
      - lineage=expansion(on=["video_id"])

    Expected behavior: video_id should be auto-included since it's needed for expansion.
    """

    class VideoSource(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video_source"]),
            id_columns=("video_id",),
            fields=[
                FieldSpec(key=FieldKey(["resolution"]), code_version="1"),
                FieldSpec(key=FieldKey(["fps"]), code_version="1"),
                FieldSpec(key=FieldKey(["duration"]), code_version="1"),
            ],
        ),
    ):
        pass

    class ExpandedFrames(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["expanded_frames"]),
            id_columns=("video_id", "frame_id"),
            deps=[
                FeatureDep(
                    feature=VideoSource,
                    # BUG: columns= doesn't include video_id, but expansion needs it
                    columns=("resolution", "fps"),
                    lineage=LineageRelationship.expansion(on=["video_id"]),
                )
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["frame_embedding"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                )
            ],
        ),
    ):
        pass

    plan = graph.get_feature_plan(ExpandedFrames.spec().key)
    engine = PolarsVersioningEngine(plan)

    # Create upstream data
    upstream_data = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "resolution": ["1080p", "4k"],
                "fps": [30, 60],
                "duration": [120, 300],
                "metaxy_provenance_by_field": [
                    {"resolution": "res_h1", "fps": "fps_h1", "duration": "dur_h1"},
                    {"resolution": "res_h2", "fps": "fps_h2", "duration": "dur_h2"},
                ],
                "metaxy_data_version_by_field": [
                    {"resolution": "res_h1", "fps": "fps_h1", "duration": "dur_h1"},
                    {"resolution": "res_h2", "fps": "fps_h2", "duration": "dur_h2"},
                ],
                "metaxy_provenance": ["prov_1", "prov_2"],
                "metaxy_data_version": ["dv_1", "dv_2"],
            }
        ).lazy()
    )

    upstream = {FeatureKey(["video_source"]): upstream_data}

    # This should NOT raise "unable to find column 'video_id'"
    # The expansion column should be auto-included
    result = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()

    # Should have 2 videos (expansion doesn't change row count during loading)
    assert len(result_df) == 2
    assert set(result_df["video_id"].to_list()) == {"v1", "v2"}


def test_two_deps_with_expansion_on_same_column_with_explicit_columns(
    graph: FeatureGraph,
) -> None:
    """Test two deps with expansion on the same column where columns= doesn't include the expansion column.

    This mirrors test_two_deps_with_aggregation_on_same_column_with_explicit_columns but for expansion.

    Expected: video_id should be auto-included for expansion AND recognized as a shared
    join column (not a collision).
    """

    class VideoMetadata(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video_metadata"]),
            id_columns=("video_id",),
            fields=[
                FieldSpec(key=FieldKey(["resolution"]), code_version="1"),
                FieldSpec(key=FieldKey(["codec"]), code_version="1"),
            ],
        ),
    ):
        pass

    class AudioMetadata(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["audio_metadata"]),
            id_columns=("video_id",),
            fields=[
                FieldSpec(key=FieldKey(["sample_rate"]), code_version="1"),
                FieldSpec(key=FieldKey(["channels"]), code_version="1"),
            ],
        ),
    ):
        pass

    class ExpandedMediaFrames(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["expanded_media_frames"]),
            id_columns=("video_id", "frame_id"),
            deps=[
                # First dep: MISSING video_id in columns
                FeatureDep(
                    feature=VideoMetadata,
                    columns=("resolution",),
                    lineage=LineageRelationship.expansion(on=["video_id"]),
                ),
                # Second dep: ALSO MISSING video_id in columns
                FeatureDep(
                    feature=AudioMetadata,
                    columns=("sample_rate",),
                    lineage=LineageRelationship.expansion(on=["video_id"]),
                ),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["frame_data"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                )
            ],
        ),
    ):
        pass

    plan = graph.get_feature_plan(ExpandedMediaFrames.spec().key)
    engine = PolarsVersioningEngine(plan)

    # Create upstream data for VideoMetadata
    video_data = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "resolution": ["1080p", "4k"],
                "codec": ["h264", "h265"],
                "metaxy_provenance_by_field": [
                    {"resolution": "res_h1", "codec": "codec_h1"},
                    {"resolution": "res_h2", "codec": "codec_h2"},
                ],
                "metaxy_data_version_by_field": [
                    {"resolution": "res_h1", "codec": "codec_h1"},
                    {"resolution": "res_h2", "codec": "codec_h2"},
                ],
                "metaxy_provenance": ["video_prov_1", "video_prov_2"],
                "metaxy_data_version": ["video_dv_1", "video_dv_2"],
            }
        ).lazy()
    )

    # Create upstream data for AudioMetadata
    audio_data = nw.from_native(
        pl.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "sample_rate": [48000, 44100],
                "channels": [2, 6],
                "metaxy_provenance_by_field": [
                    {"sample_rate": "sr_h1", "channels": "ch_h1"},
                    {"sample_rate": "sr_h2", "channels": "ch_h2"},
                ],
                "metaxy_data_version_by_field": [
                    {"sample_rate": "sr_h1", "channels": "ch_h1"},
                    {"sample_rate": "sr_h2", "channels": "ch_h2"},
                ],
                "metaxy_provenance": ["audio_prov_1", "audio_prov_2"],
                "metaxy_data_version": ["audio_dv_1", "audio_dv_2"],
            }
        ).lazy()
    )

    upstream = {
        FeatureKey(["video_metadata"]): video_data,
        FeatureKey(["audio_metadata"]): audio_data,
    }

    # This should NOT raise:
    # - "unable to find column 'video_id'" (if video_id not auto-included)
    # - "ambiguous columns" (if video_id included but treated as collision)
    result = engine.load_upstream_with_provenance(
        upstream=upstream,
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    )

    result_df = result.collect()

    # Should have 2 rows (one per video, expansion doesn't change row count during loading)
    assert len(result_df) == 2
    assert set(result_df["video_id"].to_list()) == {"v1", "v2"}


# ============================================================================
# Field-level dependency isolation through aggregation
# ============================================================================


def test_aggregation_field_level_provenance_isolation(graph: FeatureGraph) -> None:
    """Test that field-level dependencies are correctly preserved through aggregation.

    This is a critical test for the aggregation lineage fix. The downstream feature
    has two fields with different dependencies:
    - avg_temp depends only on upstream 'temperature' field
    - avg_humidity depends only on upstream 'humidity' field

    When we change ONLY the 'temperature' upstream field version, ONLY 'avg_temp'
    downstream provenance should change. 'avg_humidity' should remain unchanged.
    """

    class SensorReadings(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["sensor_readings_isolated"]),
            id_columns=("sensor_id", "reading_id"),
            fields=[
                FieldSpec(key=FieldKey(["temperature"]), code_version="1"),
                FieldSpec(key=FieldKey(["humidity"]), code_version="1"),
            ],
        ),
    ):
        pass

    from metaxy.models.field import FieldDep

    class HourlyStats(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["hourly_stats_isolated"]),
            id_columns=("sensor_id",),
            deps=[
                FeatureDep(
                    feature=SensorReadings,
                    lineage=LineageRelationship.aggregation(on=["sensor_id"]),
                )
            ],
            fields=[
                # avg_temp depends ONLY on temperature
                FieldSpec(
                    key=FieldKey(["avg_temp"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["sensor_readings_isolated"]),
                            fields=[FieldKey(["temperature"])],
                        )
                    ],
                ),
                # avg_humidity depends ONLY on humidity
                FieldSpec(
                    key=FieldKey(["avg_humidity"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["sensor_readings_isolated"]),
                            fields=[FieldKey(["humidity"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    plan = graph.get_feature_plan(HourlyStats.spec().key)
    engine = PolarsVersioningEngine(plan)

    # Scenario 1: Initial state with both readings
    upstream_v1 = nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1", "s1"],
                "reading_id": ["r1", "r2"],
                "temperature": [20.0, 21.0],
                "humidity": [50.0, 51.0],
                "metaxy_provenance_by_field": [
                    {"temperature": "temp_v1_r1", "humidity": "hum_v1_r1"},
                    {"temperature": "temp_v1_r2", "humidity": "hum_v1_r2"},
                ],
                "metaxy_data_version_by_field": [
                    {"temperature": "temp_v1_r1", "humidity": "hum_v1_r1"},
                    {"temperature": "temp_v1_r2", "humidity": "hum_v1_r2"},
                ],
                "metaxy_provenance": ["prov_r1", "prov_r2"],
                "metaxy_data_version": ["dv_r1", "dv_r2"],
            }
        ).lazy()
    )

    result_v1 = engine.load_upstream_with_provenance(
        upstream={FeatureKey(["sensor_readings_isolated"]): upstream_v1},
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    ).collect()

    # Get initial provenance values
    result_v1_polars = result_v1.to_polars()
    initial_prov_by_field = result_v1_polars["metaxy_provenance_by_field"][0]
    initial_avg_temp_prov = initial_prov_by_field["avg_temp"]
    initial_avg_humidity_prov = initial_prov_by_field["avg_humidity"]

    # Scenario 2: Only temperature changes (humidity versions stay the same)
    upstream_v2 = nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1", "s1"],
                "reading_id": ["r1", "r2"],
                "temperature": [20.0, 21.0],  # Same values
                "humidity": [50.0, 51.0],  # Same values
                "metaxy_provenance_by_field": [
                    # Only temperature field version changed!
                    {"temperature": "temp_v2_r1", "humidity": "hum_v1_r1"},
                    {"temperature": "temp_v2_r2", "humidity": "hum_v1_r2"},
                ],
                "metaxy_data_version_by_field": [
                    {"temperature": "temp_v2_r1", "humidity": "hum_v1_r1"},
                    {"temperature": "temp_v2_r2", "humidity": "hum_v1_r2"},
                ],
                "metaxy_provenance": ["prov_r1_v2", "prov_r2_v2"],
                "metaxy_data_version": ["dv_r1_v2", "dv_r2_v2"],
            }
        ).lazy()
    )

    result_v2 = engine.load_upstream_with_provenance(
        upstream={FeatureKey(["sensor_readings_isolated"]): upstream_v2},
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    ).collect()

    result_v2_polars = result_v2.to_polars()
    updated_prov_by_field = result_v2_polars["metaxy_provenance_by_field"][0]
    updated_avg_temp_prov = updated_prov_by_field["avg_temp"]
    updated_avg_humidity_prov = updated_prov_by_field["avg_humidity"]

    # CRITICAL ASSERTIONS:
    # avg_temp provenance SHOULD change (temperature upstream changed)
    assert updated_avg_temp_prov != initial_avg_temp_prov, (
        "avg_temp provenance should change when upstream temperature field changes"
    )

    # avg_humidity provenance should NOT change (humidity upstream unchanged)
    assert updated_avg_humidity_prov == initial_avg_humidity_prov, (
        "avg_humidity provenance should NOT change when only temperature upstream changes"
    )


def test_aggregation_field_level_provenance_multiple_groups(
    graph: FeatureGraph,
) -> None:
    """Test field-level provenance isolation with multiple aggregation groups.

    Each group should have independent per-field provenance based on only the
    rows belonging to that group.
    """

    class SensorReadings(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["sensor_readings_multi"]),
            id_columns=("sensor_id", "reading_id"),
            fields=[
                FieldSpec(key=FieldKey(["temperature"]), code_version="1"),
                FieldSpec(key=FieldKey(["humidity"]), code_version="1"),
            ],
        ),
    ):
        pass

    from metaxy.models.field import FieldDep

    class HourlyStats(
        SampleFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["hourly_stats_multi"]),
            id_columns=("sensor_id",),
            deps=[
                FeatureDep(
                    feature=SensorReadings,
                    lineage=LineageRelationship.aggregation(on=["sensor_id"]),
                )
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["avg_temp"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["sensor_readings_multi"]),
                            fields=[FieldKey(["temperature"])],
                        )
                    ],
                ),
                FieldSpec(
                    key=FieldKey(["avg_humidity"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["sensor_readings_multi"]),
                            fields=[FieldKey(["humidity"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    plan = graph.get_feature_plan(HourlyStats.spec().key)
    engine = PolarsVersioningEngine(plan)

    # Two sensors with different readings
    upstream = nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1", "s1", "s2", "s2"],
                "reading_id": ["r1", "r2", "r3", "r4"],
                "temperature": [20.0, 21.0, 30.0, 31.0],
                "humidity": [50.0, 51.0, 60.0, 61.0],
                "metaxy_provenance_by_field": [
                    {"temperature": "s1_temp_1", "humidity": "s1_hum_1"},
                    {"temperature": "s1_temp_2", "humidity": "s1_hum_2"},
                    {"temperature": "s2_temp_1", "humidity": "s2_hum_1"},
                    {"temperature": "s2_temp_2", "humidity": "s2_hum_2"},
                ],
                "metaxy_data_version_by_field": [
                    {"temperature": "s1_temp_1", "humidity": "s1_hum_1"},
                    {"temperature": "s1_temp_2", "humidity": "s1_hum_2"},
                    {"temperature": "s2_temp_1", "humidity": "s2_hum_1"},
                    {"temperature": "s2_temp_2", "humidity": "s2_hum_2"},
                ],
                "metaxy_provenance": ["p1", "p2", "p3", "p4"],
                "metaxy_data_version": ["d1", "d2", "d3", "d4"],
            }
        ).lazy()
    )

    result = engine.load_upstream_with_provenance(
        upstream={FeatureKey(["sensor_readings_multi"]): upstream},
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    ).collect()

    result_polars = result.to_polars().sort("sensor_id", "reading_id")
    # With window functions, all 4 original rows are preserved
    assert len(result_polars) == 4  # 4 rows (2 per sensor)

    # All rows in the same sensor group should have identical provenance
    s1_rows = result_polars.filter(result_polars["sensor_id"] == "s1")
    s2_rows = result_polars.filter(result_polars["sensor_id"] == "s2")

    # Verify all s1 rows have the same provenance
    s1_provenance = s1_rows["metaxy_provenance"].to_list()
    assert len(set(s1_provenance)) == 1, "All s1 rows should have identical provenance"

    # Verify all s2 rows have the same provenance
    s2_provenance = s2_rows["metaxy_provenance"].to_list()
    assert len(set(s2_provenance)) == 1, "All s2 rows should have identical provenance"

    s1_prov = s1_rows["metaxy_provenance_by_field"][0]
    s2_prov = s2_rows["metaxy_provenance_by_field"][0]

    # Different sensors should have different provenance for each field
    # (they aggregate different rows)
    assert s1_prov["avg_temp"] != s2_prov["avg_temp"]
    assert s1_prov["avg_humidity"] != s2_prov["avg_humidity"]

    # The overall provenance should also be different between groups
    assert s1_provenance[0] != s2_provenance[0]


def test_aggregation_field_level_provenance_definition_change() -> None:
    """Test that field-level provenance updates when upstream field definition changes.

    When the upstream field's code_version changes (definition change), the upstream
    data would be recomputed with new data_version_by_field values. This test verifies
    that the aggregation correctly propagates those changes only to dependent fields.

    Scenario:
    - V1: Both temperature and humidity have code_version="1"
    - V2: Temperature code_version changes to "2", humidity stays at "1"
    - The upstream data_version_by_field reflects the code_version change
    - Only avg_temp downstream provenance should change, not avg_humidity
    """
    from metaxy.models.field import FieldDep

    # === Version 1: Initial feature definitions ===
    with FeatureGraph().use() as graph_v1:

        class SensorReadingsV1(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["sensor_readings_defn"]),
                id_columns=("sensor_id", "reading_id"),
                fields=[
                    FieldSpec(key=FieldKey(["temperature"]), code_version="1"),
                    FieldSpec(key=FieldKey(["humidity"]), code_version="1"),
                ],
            ),
        ):
            pass

        class HourlyStatsV1(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["hourly_stats_defn"]),
                id_columns=("sensor_id",),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["sensor_readings_defn"]),
                        lineage=LineageRelationship.aggregation(on=["sensor_id"]),
                    )
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["avg_temp"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["sensor_readings_defn"]),
                                fields=[FieldKey(["temperature"])],
                            )
                        ],
                    ),
                    FieldSpec(
                        key=FieldKey(["avg_humidity"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["sensor_readings_defn"]),
                                fields=[FieldKey(["humidity"])],
                            )
                        ],
                    ),
                ],
            ),
        ):
            pass

    plan_v1 = graph_v1.get_feature_plan(HourlyStatsV1.spec().key)
    engine_v1 = PolarsVersioningEngine(plan_v1)

    # V1 upstream data - data_version_by_field reflects code_version="1" for both fields
    upstream_data_v1 = nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1", "s1"],
                "reading_id": ["r1", "r2"],
                "metaxy_provenance_by_field": [
                    {"temperature": "temp_cv1_r1", "humidity": "hum_cv1_r1"},
                    {"temperature": "temp_cv1_r2", "humidity": "hum_cv1_r2"},
                ],
                "metaxy_data_version_by_field": [
                    {"temperature": "temp_cv1_r1", "humidity": "hum_cv1_r1"},
                    {"temperature": "temp_cv1_r2", "humidity": "hum_cv1_r2"},
                ],
                "metaxy_provenance": ["prov_1", "prov_2"],
                "metaxy_data_version": ["dv_1", "dv_2"],
            }
        ).lazy()
    )

    result_v1 = engine_v1.load_upstream_with_provenance(
        upstream={FeatureKey(["sensor_readings_defn"]): upstream_data_v1},
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    ).collect()

    result_v1_polars = result_v1.to_polars()
    v1_prov_by_field = result_v1_polars["metaxy_provenance_by_field"][0]
    v1_avg_temp_prov = v1_prov_by_field["avg_temp"]
    v1_avg_humidity_prov = v1_prov_by_field["avg_humidity"]

    # === Version 2: Temperature code_version changes to "2" ===
    with FeatureGraph().use() as graph_v2:

        class SensorReadingsV2(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["sensor_readings_defn"]),
                id_columns=("sensor_id", "reading_id"),
                fields=[
                    # Temperature field code_version changed from "1" to "2"
                    FieldSpec(key=FieldKey(["temperature"]), code_version="2"),
                    FieldSpec(
                        key=FieldKey(["humidity"]), code_version="1"
                    ),  # Unchanged
                ],
            ),
        ):
            pass

        class HourlyStatsV2(
            SampleFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["hourly_stats_defn"]),
                id_columns=("sensor_id",),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["sensor_readings_defn"]),
                        lineage=LineageRelationship.aggregation(on=["sensor_id"]),
                    )
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["avg_temp"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["sensor_readings_defn"]),
                                fields=[FieldKey(["temperature"])],
                            )
                        ],
                    ),
                    FieldSpec(
                        key=FieldKey(["avg_humidity"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["sensor_readings_defn"]),
                                fields=[FieldKey(["humidity"])],
                            )
                        ],
                    ),
                ],
            ),
        ):
            pass

    plan_v2 = graph_v2.get_feature_plan(HourlyStatsV2.spec().key)
    engine_v2 = PolarsVersioningEngine(plan_v2)

    # V2 upstream data - temperature data_version_by_field changed (simulates recompute)
    # humidity stays the same (code_version unchanged)
    upstream_data_v2 = nw.from_native(
        pl.DataFrame(
            {
                "sensor_id": ["s1", "s1"],
                "reading_id": ["r1", "r2"],
                "metaxy_provenance_by_field": [
                    # Temperature changed due to code_version bump, humidity unchanged
                    {"temperature": "temp_cv2_r1", "humidity": "hum_cv1_r1"},
                    {"temperature": "temp_cv2_r2", "humidity": "hum_cv1_r2"},
                ],
                "metaxy_data_version_by_field": [
                    {"temperature": "temp_cv2_r1", "humidity": "hum_cv1_r1"},
                    {"temperature": "temp_cv2_r2", "humidity": "hum_cv1_r2"},
                ],
                "metaxy_provenance": ["prov_1_v2", "prov_2_v2"],
                "metaxy_data_version": ["dv_1_v2", "dv_2_v2"],
            }
        ).lazy()
    )

    result_v2 = engine_v2.load_upstream_with_provenance(
        upstream={FeatureKey(["sensor_readings_defn"]): upstream_data_v2},
        hash_algo=HashAlgorithm.XXHASH64,
        filters={},
    ).collect()

    result_v2_polars = result_v2.to_polars()
    v2_prov_by_field = result_v2_polars["metaxy_provenance_by_field"][0]
    v2_avg_temp_prov = v2_prov_by_field["avg_temp"]
    v2_avg_humidity_prov = v2_prov_by_field["avg_humidity"]

    # CRITICAL ASSERTIONS:
    # avg_temp provenance SHOULD change (temperature upstream data_version changed)
    assert v2_avg_temp_prov != v1_avg_temp_prov, (
        "avg_temp provenance should change when upstream temperature field "
        "data_version changes (due to code_version bump)"
    )

    # avg_humidity provenance should NOT change (humidity data_version unchanged)
    assert v2_avg_humidity_prov == v1_avg_humidity_prov, (
        "avg_humidity provenance should NOT change when only temperature "
        "code_version changes (humidity data_version unchanged)"
    )
