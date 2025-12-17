"""Test metadata store provenance calculation against golden reference.

This file tests the CORRECTNESS of provenance calculations by comparing store
implementations against a golden reference. It focuses on:
- Verifying stores produce correct provenance (matches golden reference)
- Testing deduplication logic (keep_latest_by_group)
- Testing edge cases (duplicates, partial duplicates, etc.)

Hash algorithm and truncation testing is handled in test_hash_algorithms.py.
Store-specific behavior testing is handled in test_resolve_update.py.

The goal here is to verify that the core provenance calculation is correct,
not to test every possible combination of hash algorithm × store × truncation.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, TypeAlias

import polars as pl
import polars.testing as pl_testing
import pytest
import pytest_cases
from hypothesis.errors import NonInteractiveExampleWarning
from pytest_cases import parametrize_with_cases

from metaxy import (
    BaseFeature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FieldSpec,
    LineageRelationship,
)
from metaxy._testing.models import SampleFeature, SampleFeatureSpec
from metaxy._testing.parametric import downstream_metadata_strategy
from metaxy._utils import collect_to_polars
from metaxy.metadata_store import (
    HashAlgorithmNotSupportedError,
    MetadataStore,
)
from metaxy.models.field import SpecialFieldDep
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FieldKey

if TYPE_CHECKING:
    pass

FeaturePlanOutput: TypeAlias = tuple[
    FeatureGraph, Mapping[FeatureKey, type[BaseFeature]], FeaturePlan
]


class FeaturePlanCases:
    """Test cases for different feature plan configurations."""

    def case_single_upstream(self, graph: FeatureGraph) -> FeaturePlanOutput:
        """Simple parent->child feature plan with single upstream."""

        class ParentFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="parent",
                fields=["foo"],
            ),
        ):
            sample_uid: str

        class ChildFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="child",
                deps=[FeatureDep(feature=ParentFeature)],
                fields=["foo"],
            ),
        ):
            pass

        # Get the feature plan for child
        child_plan = graph.get_feature_plan(ChildFeature.spec().key)

        upstream_features = {
            ParentFeature.spec().key: ParentFeature,
        }

        return graph, upstream_features, child_plan

    def case_two_upstreams(self, graph: FeatureGraph) -> FeaturePlanOutput:
        """Feature plan with two upstream dependencies."""

        class Parent1Feature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="parent1",
                fields=["foo"],
            ),
        ):
            sample_uid: str

        class Parent2Feature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="parent2",
                fields=["foo"],
            ),
        ):
            sample_uid: str

        class ChildFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="child",
                deps=[
                    FeatureDep(feature=Parent1Feature),
                    FeatureDep(feature=Parent2Feature),
                ],
                fields=["foo"],
            ),
        ):
            pass

        # Get the feature plan for child
        child_plan = graph.get_feature_plan(ChildFeature.spec().key)

        upstream_features = {
            Parent1Feature.spec().key: Parent1Feature,
            Parent2Feature.spec().key: Parent2Feature,
        }

        return graph, upstream_features, child_plan


# Removed: TruncationCases and metaxy_config fixture
# Hash truncation is now tested in test_hash_algorithms.py


def setup_store_with_data(
    empty_store: MetadataStore,
    feature_plan_config: FeaturePlanOutput,
) -> tuple[MetadataStore, FeaturePlanOutput, pl.DataFrame]:
    # Unpack feature plan configuration
    graph, upstream_features, child_feature_plan = feature_plan_config

    # Get the child feature from the graph
    child_key = child_feature_plan.feature.key
    ChildFeature = graph.features_by_key[child_key]

    # Feature versions for strategy
    child_version = ChildFeature.feature_version()

    feature_versions = {}
    for feature_key, upstream_feature in upstream_features.items():
        feature_versions[feature_key.to_string()] = upstream_feature.feature_version()
    feature_versions[child_key.to_string()] = child_version

    # Generate test data using the golden strategy
    # Note: Using .example() in test infrastructure is appropriate for generating
    # deterministic test data with pytest-cases parametrization. We suppress the
    # NonInteractiveExampleWarning since this is not interactive exploration.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        example_data = downstream_metadata_strategy(
            child_feature_plan,
            feature_versions=feature_versions,
            snapshot_version=graph.snapshot_version,
            hash_algorithm=empty_store.hash_algorithm,
            min_rows=5,
            max_rows=20,
        ).example()

    upstream_data, golden_downstream = example_data

    # Write upstream metadata to store
    try:
        with empty_store:
            for feature_key, upstream_feature in upstream_features.items():
                upstream_df = upstream_data[feature_key.to_string()]
                empty_store.write_metadata(upstream_feature, upstream_df)
    except HashAlgorithmNotSupportedError:
        pytest.skip(
            f"Hash algorithm {empty_store.hash_algorithm} not supported by {empty_store}"
        )

    return empty_store, feature_plan_config, golden_downstream


# Removed: EmptyStoreCases with hash algorithm parametrization
# Now using simplified fixtures from conftest.py
# Hash algorithm × store combinations are tested in test_hash_algorithms.py


@parametrize_with_cases("feature_plan_config", cases=FeaturePlanCases)
def test_store_resolve_update_matches_golden_provenance(
    any_store: MetadataStore,
    feature_plan_config: FeaturePlanOutput,
):
    """Test metadata store provenance calculation matches golden reference."""
    empty_store = any_store
    # Setup store with upstream data and get golden reference
    store, (graph, upstream_features, child_feature_plan), golden_downstream = (
        setup_store_with_data(
            empty_store,
            feature_plan_config,
        )
    )

    # Get the child feature from the graph
    child_key = child_feature_plan.feature.key
    ChildFeature = graph.features_by_key[child_key]

    # Get child version
    child_version = ChildFeature.feature_version()

    # Call resolve_update to compute provenance
    with store:
        try:
            increment = store.resolve_update(
                ChildFeature,
                target_version=child_version,
                snapshot_version=graph.snapshot_version,
            )
        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {store.hash_algorithm} not supported by {store}"
            )

        added_df = increment.added.lazy().collect().to_polars()
        # Get ID columns from the feature plan
        id_columns = list(child_feature_plan.feature.id_columns)
        # Sort both DataFrames by ID columns for comparison
        added_sorted = added_df.sort(id_columns)
        golden_sorted = golden_downstream.sort(id_columns)

        # Select only the columns that exist in both (resolve_update may not return all metadata columns)
        # Exclude metaxy_created_at since it's a timestamp that differs between generation times
        from metaxy.models.constants import METAXY_CREATED_AT

        common_columns = [
            col
            for col in added_sorted.columns
            if col in golden_sorted.columns and col != METAXY_CREATED_AT
        ]
        added_selected = added_sorted.select(common_columns)
        golden_selected = golden_sorted.select(common_columns)

        # Use Polars testing utility to compare DataFrames
        # This will check all columns including the provenance_by_field struct
        pl_testing.assert_frame_equal(
            added_selected,
            golden_selected,
            check_row_order=True,
            check_column_order=False,
        )


# ============= TEST: DEDUPLICATION WITH DUPLICATES =============


@parametrize_with_cases("feature_plan_config", cases=FeaturePlanCases)
def test_golden_reference_with_duplicate_timestamps(
    any_store: MetadataStore,
    feature_plan_config: FeaturePlanOutput,
):
    """Test deduplication logic correctly filters older versions before computing provenance."""
    empty_store = any_store
    # Setup store with upstream data and get golden reference
    store, (graph, upstream_features, child_feature_plan), golden_downstream = (
        setup_store_with_data(
            empty_store,
            feature_plan_config,
        )
    )

    # Get the child feature from the graph
    child_key = child_feature_plan.feature.key
    ChildFeature = graph.features_by_key[child_key]
    child_version = ChildFeature.feature_version()

    with store:
        try:
            from datetime import timedelta

            import polars as pl

            from metaxy.models.constants import METAXY_CREATED_AT

            # Add older duplicates to upstream metadata
            for feature_key, upstream_feature in upstream_features.items():
                # Read existing upstream data
                existing_df = (
                    store.read_metadata(upstream_feature).lazy().collect().to_polars()
                )

                # Create older duplicates (same IDs, older timestamps)
                older_df = existing_df.clone()
                older_df = older_df.with_columns(
                    (pl.col(METAXY_CREATED_AT) - timedelta(hours=2)).alias(
                        METAXY_CREATED_AT
                    )
                )

                # Modify a field value to ensure different provenance
                # This tests that older version is NOT used
                user_fields = [
                    col
                    for col in older_df.columns
                    if not col.startswith("metaxy_") and col != "sample_uid"
                ]
                if user_fields:
                    field = user_fields[0]
                    older_df = older_df.with_columns(
                        pl.when(pl.col(field).is_not_null())
                        .then(pl.col(field).cast(pl.Utf8) + "_DUPLICATE_OLD")
                        .otherwise(pl.col(field))
                        .alias(field)
                    )

                # Write the older duplicates
                store.write_metadata(upstream_feature, older_df)

                # Now store has 2 versions per sample:
                # - Original (newer) - should be used
                # - Duplicate (older, modified) - should be ignored

            # Call resolve_update - should use only latest versions
            increment = store.resolve_update(
                ChildFeature,
                target_version=child_version,
                snapshot_version=graph.snapshot_version,
            )

        except HashAlgorithmNotSupportedError:
            pytest.skip(
                f"Hash algorithm {store.hash_algorithm} not supported by {store}"
            )

        added_df = increment.added.lazy().collect().to_polars()

        # Get ID columns from the feature plan
        id_columns = list(child_feature_plan.feature.id_columns)

        # Sort both DataFrames by ID columns for comparison
        added_sorted = added_df.sort(id_columns)
        golden_sorted = golden_downstream.sort(id_columns)

        # Exclude metaxy_created_at since it's a timestamp
        common_columns = [
            col
            for col in added_sorted.columns
            if col in golden_sorted.columns and col != METAXY_CREATED_AT
        ]
        added_selected = added_sorted.select(common_columns)
        golden_selected = golden_sorted.select(common_columns)

        # Verify that computed provenance matches golden reference
        # This proves deduplication worked correctly - only latest versions were used
        pl_testing.assert_frame_equal(
            added_selected,
            golden_selected,
            check_row_order=True,
            check_column_order=False,
        )


def test_golden_reference_with_all_duplicates_same_timestamp(
    any_store: MetadataStore,
    graph: FeatureGraph,
):
    """Test deduplication with all samples having duplicate entries at same timestamp."""
    empty_store = any_store

    # Create simple feature graph
    class ParentFeature(
        SampleFeature,
        spec=SampleFeatureSpec(
            key="parent",
            fields=["value"],
        ),
    ):
        pass

    class ChildFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="child",
            deps=[FeatureDep(feature=ParentFeature)],
            fields=["computed"],
        ),
    ):
        pass

    child_plan = graph.get_feature_plan(ChildFeature.spec().key)

    # Generate golden reference data
    feature_versions = {
        "parent": ParentFeature.feature_version(),
        "child": ChildFeature.feature_version(),
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        upstream_data, golden_downstream = downstream_metadata_strategy(
            child_plan,
            feature_versions=feature_versions,
            snapshot_version=graph.snapshot_version,
            hash_algorithm=empty_store.hash_algorithm,
            min_rows=5,
            max_rows=10,
        ).example()

    parent_df = upstream_data["parent"]

    try:
        with empty_store:
            from datetime import datetime

            import polars as pl

            from metaxy.models.constants import METAXY_CREATED_AT

            # Create duplicates with SAME timestamp for ALL samples
            same_timestamp = datetime.now()

            # Write first version
            version1 = parent_df.with_columns(
                pl.lit(same_timestamp).alias(METAXY_CREATED_AT)
            )
            empty_store.write_metadata(ParentFeature, version1)

            # Create second version with same timestamp but different provenance
            # We modify the provenance columns to simulate different data
            version2 = parent_df.with_columns(
                pl.lit(same_timestamp).alias(METAXY_CREATED_AT),
                # Modify provenance to make it different (simulate different underlying data)
                (pl.col("metaxy_provenance").cast(pl.Utf8) + "_DUP").alias(
                    "metaxy_provenance"
                ),
            )

            # Write duplicates with same timestamp
            empty_store.write_metadata(ParentFeature, version2)

            # Now every sample has 2 versions with same timestamp
            # Call resolve_update - should pick one deterministically
            increment = empty_store.resolve_update(
                ChildFeature,
                target_version=ChildFeature.feature_version(),
                snapshot_version=graph.snapshot_version,
            )

            # Verify we got results (deterministic even with same timestamps)
            added_df = increment.added.lazy().collect().to_polars()
            assert len(added_df) > 0, (
                "Expected at least some samples after deduplication"
            )

            # With duplicates at same timestamp, we should still get the original count
            # (deduplication picks one version per sample)
            assert len(added_df) == len(parent_df), (
                f"Expected {len(parent_df)} deduplicated samples, got {len(added_df)}"
            )

    except HashAlgorithmNotSupportedError:
        pytest.skip(
            f"Hash algorithm {empty_store.hash_algorithm} not supported by {empty_store}"
        )


@parametrize_with_cases("feature_plan_config", cases=FeaturePlanCases)
def test_golden_reference_partial_duplicates(
    any_store: MetadataStore,
    feature_plan_config: FeaturePlanOutput,
):
    """Test golden reference with only some upstream samples having duplicates."""
    empty_store = any_store
    # Setup store with upstream data
    store, (graph, upstream_features, child_feature_plan), golden_downstream = (
        setup_store_with_data(
            empty_store,
            feature_plan_config,
        )
    )

    child_key = child_feature_plan.feature.key
    ChildFeature = graph.features_by_key[child_key]
    child_version = ChildFeature.feature_version()

    try:
        with store:
            from datetime import timedelta

            import polars as pl

            from metaxy.models.constants import METAXY_CREATED_AT

            # Add older duplicates for only HALF of the samples in each upstream
            for feature_key, upstream_feature in upstream_features.items():
                existing_df = (
                    store.read_metadata(upstream_feature).lazy().collect().to_polars()
                )

                # Get half of samples
                num_samples = len(existing_df)
                half_count = num_samples // 2

                if half_count > 0:
                    # Take first half
                    samples_to_duplicate = existing_df.head(half_count)

                    # Create older version
                    older_df = samples_to_duplicate.with_columns(
                        (pl.col(METAXY_CREATED_AT) - timedelta(hours=1)).alias(
                            METAXY_CREATED_AT
                        )
                    )

                    # Write older duplicates
                    store.write_metadata(upstream_feature, older_df)

                # Now store has:
                # - First half of samples: 2 versions each (newer and older)
                # - Second half of samples: 1 version each (original only)

            # Call resolve_update
            increment = store.resolve_update(
                ChildFeature,
                target_version=child_version,
                snapshot_version=graph.snapshot_version,
            )

            added_df = increment.added.lazy().collect().to_polars()
            id_columns = list(child_feature_plan.feature.id_columns)

            # Sort both for comparison
            added_sorted = added_df.sort(id_columns)
            golden_sorted = golden_downstream.sort(id_columns)

            # Exclude timestamp
            common_columns = [
                col
                for col in added_sorted.columns
                if col in golden_sorted.columns and col != METAXY_CREATED_AT
            ]
            added_selected = added_sorted.select(common_columns)
            golden_selected = golden_sorted.select(common_columns)

            # Verify provenance matches golden reference
            pl_testing.assert_frame_equal(
                added_selected,
                golden_selected,
                check_row_order=True,
                check_column_order=False,
            )

    except HashAlgorithmNotSupportedError:
        pytest.skip(f"Hash algorithm {store.hash_algorithm} not supported by {store}")


# ============= UNIT TEST: keep_latest_by_group =============


class KeepLatestTestDataCases:
    """Test data cases for keep_latest_by_group tests."""

    def case_polars(self):
        import narwhals as nw

        from metaxy.versioning.polars import PolarsVersioningEngine

        base_time = datetime(2024, 1, 1, 12, 0, 0)

        def create_data_fn(pl_df):
            return nw.from_native(pl_df)

        return (PolarsVersioningEngine, create_data_fn, base_time)

    def case_ibis(self, tmp_path):
        import ibis
        import narwhals as nw

        from metaxy.versioning.ibis import IbisVersioningEngine

        base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Create a persistent connection for this test case
        con = ibis.duckdb.connect(tmp_path / "test.duckdb")
        table_counter = [0]  # Mutable counter for unique table names

        def create_data_fn(pl_df):
            # Create a unique table name for this invocation
            table_counter[0] += 1
            table_name = f"test_data_{table_counter[0]}"

            # Write to DuckDB and return as Narwhals-wrapped Ibis table
            con.create_table(table_name, pl_df.to_pandas(), overwrite=True)
            ibis_table = con.table(table_name)
            return nw.from_native(ibis_table)

        return (IbisVersioningEngine, create_data_fn, base_time)


@pytest_cases.fixture
@parametrize_with_cases("test_data", cases=KeepLatestTestDataCases)
def keep_latest_test_data(test_data):
    return test_data


def test_keep_latest_by_group(keep_latest_test_data):
    from datetime import timedelta

    import polars as pl

    # Get fixture data
    engine_class, create_data_fn, base_time = keep_latest_test_data

    # Create test data with 5 versions of the same sample
    data = pl.DataFrame(
        {
            "sample_uid": ["sample1"] * 5,
            "value": [10, 20, 30, 40, 50],  # Different values per version
            "timestamp": [
                base_time + timedelta(hours=i) for i in range(5)
            ],  # Increasing timestamps
        }
    )

    # Shuffle to ensure order doesn't matter
    shuffled_data = data.sample(fraction=1.0, shuffle=True, seed=42)

    # Convert to Narwhals using the case-specific function
    nw_df = create_data_fn(shuffled_data)

    # Call keep_latest_by_group directly (staticmethod)
    result_nw = engine_class.keep_latest_by_group(
        nw_df,
        group_columns=["sample_uid"],
        timestamp_column="timestamp",
    )

    # Convert result to Polars for assertion
    result = collect_to_polars(result_nw)

    # Verify we got exactly 1 row (only the latest)
    assert len(result) == 1, f"Expected 1 row, got {len(result)}"

    # Verify it's the latest version (value=50)
    assert result["value"][0] == 50, (
        f"Expected value=50 (latest), got {result['value'][0]}"
    )

    # Verify the timestamp is the latest
    expected_timestamp = base_time + timedelta(hours=4)
    assert result["timestamp"][0] == expected_timestamp, (
        f"Expected timestamp={expected_timestamp}, got {result['timestamp'][0]}"
    )


def test_keep_latest_by_group_aggregation_n_to_1(keep_latest_test_data):
    """Test keep_latest_by_group with N:1 aggregation (sensor readings to hourly stats)."""
    from datetime import timedelta

    import polars as pl

    engine_class, create_data_fn, base_time = keep_latest_test_data

    # Create sensor readings with duplicates (multiple versions of same reading)
    # reading_id identifies individual readings, but we have 2 versions of each
    data = pl.DataFrame(
        {
            "sensor_id": ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"],
            "hour": ["10h"] * 8,
            "reading_id": [
                "r1",
                "r1",
                "r2",
                "r2",
                "r3",
                "r3",
                "r4",
                "r4",
            ],  # Duplicates
            "temperature": [
                20.0,
                20.5,
                21.0,
                21.5,
                19.0,
                19.5,
                22.0,
                22.5,
            ],  # Different values
            "timestamp": [
                base_time + timedelta(hours=i) for i in range(8)
            ],  # Increasing timestamps
        }
    )

    # Shuffle to ensure order doesn't matter
    shuffled_data = data.sample(fraction=1.0, shuffle=True, seed=42)

    # Convert to Narwhals using the case-specific function
    nw_df = create_data_fn(shuffled_data)

    # Call keep_latest_by_group
    result_nw = engine_class.keep_latest_by_group(
        nw_df,
        group_columns=["sensor_id", "hour", "reading_id"],
        timestamp_column="timestamp",
    )

    # Convert result to Polars for assertion
    result = collect_to_polars(result_nw)

    # Verify we got exactly 4 rows (one per reading_id: r1, r2, r3, r4)
    assert len(result) == 4, f"Expected 4 rows (one per reading), got {len(result)}"

    # Verify only latest versions kept (the ones with higher temperature values)
    result_sorted = result.sort(["sensor_id", "reading_id"])
    assert result_sorted["temperature"].to_list() == [20.5, 21.5, 19.5, 22.5], (
        "Expected latest versions with higher temperatures"
    )


def test_keep_latest_by_group_expansion_1_to_n(keep_latest_test_data):
    """Test keep_latest_by_group with 1:N expansion (video to video frames)."""
    from datetime import timedelta

    import polars as pl

    engine_class, create_data_fn, base_time = keep_latest_test_data

    # Create video metadata with duplicates (old and new versions)
    # Same video_id but different metadata versions
    data = pl.DataFrame(
        {
            "video_id": ["v1", "v1", "v1", "v2", "v2"],  # Duplicates for each video
            "resolution": ["720p", "1080p", "4K", "720p", "1080p"],  # Different values
            "fps": [30, 30, 60, 30, 60],  # Different values
            "timestamp": [
                base_time + timedelta(hours=i) for i in range(5)
            ],  # Increasing timestamps
        }
    )

    # Shuffle to ensure order doesn't matter
    shuffled_data = data.sample(fraction=1.0, shuffle=True, seed=42)

    # Convert to Narwhals using the case-specific function
    nw_df = create_data_fn(shuffled_data)

    # Call keep_latest_by_group
    result_nw = engine_class.keep_latest_by_group(
        nw_df,
        group_columns=["video_id"],
        timestamp_column="timestamp",
    )

    # Convert result to Polars for assertion
    result = collect_to_polars(result_nw)

    # Verify we got exactly 2 rows (one per video_id: v1, v2)
    assert len(result) == 2, f"Expected 2 rows (one per video), got {len(result)}"

    # Verify only latest versions kept
    result_sorted = result.sort("video_id")

    # v1's latest version is "4K" (3rd occurrence, timestamp +2 hours)
    # v2's latest version is "1080p" (2nd occurrence, timestamp +4 hours)
    assert result_sorted["resolution"].to_list() == ["4K", "1080p"], (
        "Expected latest versions: v1=4K, v2=1080p"
    )
    assert result_sorted["fps"].to_list() == [60, 60], "Expected latest fps values"


# ============= MIXED LINEAGE RESOLVE_UPDATE TESTS =============


def test_resolve_update_aggregation_plus_identity(
    any_store: MetadataStore,
    graph: FeatureGraph,
    snapshot,
):
    """Test resolve_update with mixed aggregation + identity dependencies.

    Scenario:
    - SensorReadings: N:1 aggregation to hourly stats
    - SensorInfo: 1:1 identity join with sensor metadata

    Both upstream features have the same ID columns (sensor_id) to allow joining.
    The aggregation reduces multiple readings per sensor to one per sensor.

    Expected: Downstream provenance combines aggregated readings provenance
    and identity sensor info provenance.
    """

    class SensorInfo(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="sensor_info",
            id_columns=("sensor_id",),
            fields=[
                FieldSpec(key=FieldKey(["location"]), code_version="1"),
            ],
        ),
    ):
        sensor_id: str

    class SensorReadings(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="sensor_readings",
            id_columns=("sensor_id", "reading_id"),
            fields=[
                FieldSpec(key=FieldKey(["temperature"]), code_version="1"),
            ],
        ),
    ):
        sensor_id: str
        reading_id: str

    class AggregatedSensorStats(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="aggregated_sensor_stats",
            id_columns=("sensor_id",),
            deps=[
                FeatureDep(
                    feature=SensorReadings,
                    lineage=LineageRelationship.aggregation(on=["sensor_id"]),
                ),
                FeatureDep(
                    feature=SensorInfo,
                    lineage=LineageRelationship.identity(),
                ),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["enriched_stat"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                ),
            ],
        ),
    ):
        sensor_id: str

    try:
        with any_store:
            # === INITIAL DATA ===
            # Write sensor info (identity upstream)
            sensor_info_df = pl.DataFrame(
                {
                    "sensor_id": ["s1", "s2"],
                    "metaxy_provenance_by_field": [
                        {"location": "loc_hash_1"},
                        {"location": "loc_hash_2"},
                    ],
                }
            )
            any_store.write_metadata(SensorInfo, sensor_info_df)

            # Write sensor readings (aggregation upstream) - multiple per sensor
            readings_df = pl.DataFrame(
                {
                    "sensor_id": ["s1", "s1", "s1", "s2", "s2"],
                    "reading_id": ["r1", "r2", "r3", "r4", "r5"],
                    "metaxy_provenance_by_field": [
                        {"temperature": "temp_hash_1"},
                        {"temperature": "temp_hash_2"},
                        {"temperature": "temp_hash_3"},
                        {"temperature": "temp_hash_4"},
                        {"temperature": "temp_hash_5"},
                    ],
                }
            )
            any_store.write_metadata(SensorReadings, readings_df)

            # === FIRST RESOLVE UPDATE ===
            increment = any_store.resolve_update(
                AggregatedSensorStats,
                target_version=AggregatedSensorStats.feature_version(),
                snapshot_version=graph.snapshot_version,
            )

            added_df = increment.added.lazy().collect().to_polars()

            # Check row count: aggregation reduces 5 readings to 2 (one per sensor)
            assert len(added_df) == 2, (
                f"Expected 2 rows (one per sensor), got {len(added_df)}"
            )

            # Check ID columns are correct
            assert "sensor_id" in added_df.columns, "Missing sensor_id column"
            assert set(added_df["sensor_id"].to_list()) == {"s1", "s2"}, (
                "Unexpected sensor IDs"
            )

            # Check provenance columns exist
            assert "metaxy_provenance" in added_df.columns, "Missing metaxy_provenance"
            assert "metaxy_provenance_by_field" in added_df.columns, (
                "Missing metaxy_provenance_by_field"
            )

            # Verify provenance_by_field has the expected structure
            # It should contain a hash for "enriched_stat" field
            for prov_by_field in added_df["metaxy_provenance_by_field"]:
                assert "enriched_stat" in prov_by_field, (
                    f"Expected 'enriched_stat' field in provenance_by_field, got: {prov_by_field}"
                )

            # Write downstream metadata so next resolve_update can detect changes
            any_store.write_metadata(AggregatedSensorStats, added_df)

            # === SECOND RESOLVE UPDATE (no changes) ===
            increment_no_change = any_store.resolve_update(
                AggregatedSensorStats,
                target_version=AggregatedSensorStats.feature_version(),
                snapshot_version=graph.snapshot_version,
            )
            added_no_change = increment_no_change.added.lazy().collect().to_polars()

            # No changes expected - upstream hasn't changed
            assert len(added_no_change) == 0, (
                f"Expected 0 rows when nothing changed, got {len(added_no_change)}"
            )

            # Snapshot the state for regression testing
            final_df = (
                any_store.read_metadata(AggregatedSensorStats)
                .lazy()
                .collect()
                .to_polars()
            )
            final_sorted = final_df.sort("sensor_id")
            snapshot_data = [
                {
                    "sensor_id": final_sorted["sensor_id"][i],
                    "metaxy_provenance_by_field": final_sorted[
                        "metaxy_provenance_by_field"
                    ][i],
                }
                for i in range(len(final_sorted))
            ]
            assert snapshot_data == snapshot

    except HashAlgorithmNotSupportedError:
        pytest.skip(
            f"Hash algorithm {any_store.hash_algorithm} not supported by {any_store}"
        )


def test_resolve_update_expansion_plus_identity(
    any_store: MetadataStore,
    graph: FeatureGraph,
    snapshot,
):
    """Test resolve_update with mixed expansion + identity dependencies.

    Scenario:
    - Video: 1:N expansion (one video → many frames)
    - VideoMetadata: 1:1 identity join with video metadata

    Both features have video_id as ID column. The expansion indicates that
    downstream will have more rows than upstream Video, but change detection
    happens at the parent (video_id) level.

    Expected: Downstream provenance combines expansion parent provenance
    and identity metadata provenance at the parent level.
    """

    class Video(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="video",
            id_columns=("video_id",),
            fields=[
                FieldSpec(key=FieldKey(["content"]), code_version="1"),
            ],
        ),
    ):
        video_id: str

    class VideoMetadata(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="video_metadata",
            id_columns=("video_id",),
            fields=[
                FieldSpec(key=FieldKey(["category"]), code_version="1"),
            ],
        ),
    ):
        video_id: str

    class EnrichedFrames(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="enriched_frames",
            id_columns=("video_id", "frame_id"),  # Expanded: parent ID + frame ID
            deps=[
                FeatureDep(
                    feature=Video,
                    lineage=LineageRelationship.expansion(on=["video_id"]),
                ),
                FeatureDep(
                    feature=VideoMetadata,
                    lineage=LineageRelationship.identity(),
                ),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["frame_feature"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                ),
            ],
        ),
    ):
        video_id: str
        frame_id: str

    try:
        with any_store:
            # === INITIAL DATA ===
            # Write video (expansion upstream)
            video_df = pl.DataFrame(
                {
                    "video_id": ["v1", "v2"],
                    "metaxy_provenance_by_field": [
                        {"content": "content_hash_1"},
                        {"content": "content_hash_2"},
                    ],
                }
            )
            any_store.write_metadata(Video, video_df)

            # Write video metadata (identity upstream)
            metadata_df = pl.DataFrame(
                {
                    "video_id": ["v1", "v2"],
                    "metaxy_provenance_by_field": [
                        {"category": "cat_hash_1"},
                        {"category": "cat_hash_2"},
                    ],
                }
            )
            any_store.write_metadata(VideoMetadata, metadata_df)

            # === FIRST RESOLVE UPDATE ===
            # resolve_update returns data at parent level (video_id) for change detection
            increment = any_store.resolve_update(
                EnrichedFrames,
                target_version=EnrichedFrames.feature_version(),
                snapshot_version=graph.snapshot_version,
            )

            added_df = increment.added.lazy().collect().to_polars()

            # Check row count: resolve_update returns parent-level rows (one per video)
            assert len(added_df) == 2, (
                f"Expected 2 rows (one per video), got {len(added_df)}"
            )

            # Check parent ID column is correct
            assert "video_id" in added_df.columns, "Missing video_id column"
            assert set(added_df["video_id"].to_list()) == {"v1", "v2"}, (
                "Unexpected video IDs"
            )

            # Check provenance columns exist
            assert "metaxy_provenance" in added_df.columns, "Missing metaxy_provenance"
            assert "metaxy_provenance_by_field" in added_df.columns, (
                "Missing metaxy_provenance_by_field"
            )

            # Verify provenance_by_field has the expected structure
            for prov_by_field in added_df["metaxy_provenance_by_field"]:
                assert "frame_feature" in prov_by_field, (
                    f"Expected 'frame_feature' field in provenance_by_field, got: {prov_by_field}"
                )

            # === EXPAND: User code creates multiple frames per video ===
            # Each video expands to 3 frames, all sharing the same provenance
            expanded_rows = []
            for row in added_df.iter_rows(named=True):
                video_id = row["video_id"]
                provenance = row["metaxy_provenance"]
                provenance_by_field = row["metaxy_provenance_by_field"]

                # Create 3 frames per video
                for frame_idx in range(3):
                    expanded_rows.append(
                        {
                            "video_id": video_id,
                            "frame_id": f"{video_id}_frame_{frame_idx}",
                            "metaxy_provenance": provenance,
                            "metaxy_provenance_by_field": provenance_by_field,
                        }
                    )

            expanded_df = pl.DataFrame(expanded_rows)

            # Write expanded downstream metadata
            any_store.write_metadata(EnrichedFrames, expanded_df)

            # Verify we wrote 6 rows (2 videos × 3 frames)
            stored_df = (
                any_store.read_metadata(EnrichedFrames).lazy().collect().to_polars()
            )
            assert len(stored_df) == 6, (
                f"Expected 6 expanded rows, got {len(stored_df)}"
            )

            # === SECOND RESOLVE UPDATE (no changes) ===
            increment_no_change = any_store.resolve_update(
                EnrichedFrames,
                target_version=EnrichedFrames.feature_version(),
                snapshot_version=graph.snapshot_version,
            )
            added_no_change = increment_no_change.added.lazy().collect().to_polars()

            # No changes expected - upstream hasn't changed
            assert len(added_no_change) == 0, (
                f"Expected 0 rows when nothing changed, got {len(added_no_change)}"
            )

            # Snapshot the state for regression testing (at parent level for determinism)
            final_df = (
                any_store.read_metadata(EnrichedFrames).lazy().collect().to_polars()
            )
            # Group by video_id to get unique provenance per video
            final_grouped = (
                final_df.group_by("video_id")
                .agg(pl.col("metaxy_provenance_by_field").first())
                .sort("video_id")
            )
            provenance_data = [
                {
                    "video_id": final_grouped["video_id"][i],
                    "metaxy_provenance_by_field": final_grouped[
                        "metaxy_provenance_by_field"
                    ][i],
                }
                for i in range(len(final_grouped))
            ]
            assert provenance_data == snapshot

    except HashAlgorithmNotSupportedError:
        pytest.skip(
            f"Hash algorithm {any_store.hash_algorithm} not supported by {any_store}"
        )


def test_resolve_update_aggregation_expansion(
    any_store: MetadataStore,
    graph: FeatureGraph,
    snapshot,
):
    """Test resolve_update with aggregation + expansion dependencies.

    Scenario with two upstream features sharing config_id:
    - SensorReadings: N:1 aggregation by config_id (many readings → one per config)
    - VideoSource: 1:N expansion by config_id (one video expands to many frames)

    This tests aggregation and expansion lineage types together.

    Expected: Downstream provenance combines aggregation and expansion provenances.
    """

    class SensorReadings(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="sensor_readings_v2",
            id_columns=("config_id", "reading_id"),
            fields=[
                FieldSpec(key=FieldKey(["measurement"]), code_version="1"),
            ],
        ),
    ):
        config_id: str
        reading_id: str

    class VideoSource(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="video_source",
            id_columns=("config_id",),
            fields=[
                FieldSpec(key=FieldKey(["video_data"]), code_version="1"),
            ],
        ),
    ):
        config_id: str

    class CombinedFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="combined_feature",
            id_columns=("config_id", "output_id"),  # Expanded: parent ID + output ID
            deps=[
                FeatureDep(
                    feature=SensorReadings,
                    lineage=LineageRelationship.aggregation(on=["config_id"]),
                ),
                FeatureDep(
                    feature=VideoSource,
                    lineage=LineageRelationship.expansion(on=["config_id"]),
                ),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["combined_output"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                ),
            ],
        ),
    ):
        config_id: str
        output_id: str

    try:
        with any_store:
            # === INITIAL DATA ===
            # Write sensor readings (aggregation upstream) - multiple per config
            readings_df = pl.DataFrame(
                {
                    "config_id": ["cfg1", "cfg1", "cfg1", "cfg2", "cfg2"],
                    "reading_id": ["r1", "r2", "r3", "r4", "r5"],
                    "metaxy_provenance_by_field": [
                        {"measurement": "meas_hash_1"},
                        {"measurement": "meas_hash_2"},
                        {"measurement": "meas_hash_3"},
                        {"measurement": "meas_hash_4"},
                        {"measurement": "meas_hash_5"},
                    ],
                }
            )
            any_store.write_metadata(SensorReadings, readings_df)

            # Write video source (expansion upstream)
            video_df = pl.DataFrame(
                {
                    "config_id": ["cfg1", "cfg2"],
                    "metaxy_provenance_by_field": [
                        {"video_data": "video_hash_1"},
                        {"video_data": "video_hash_2"},
                    ],
                }
            )
            any_store.write_metadata(VideoSource, video_df)

            # === FIRST RESOLVE UPDATE ===
            # resolve_update returns data at parent level (config_id) for change detection
            increment = any_store.resolve_update(
                CombinedFeature,
                target_version=CombinedFeature.feature_version(),
                snapshot_version=graph.snapshot_version,
            )

            added_df = increment.added.lazy().collect().to_polars()

            # Check row count: resolve_update returns parent-level rows (one per config)
            assert len(added_df) == 2, (
                f"Expected 2 rows (one per config), got {len(added_df)}"
            )

            # Check parent ID column is correct
            assert "config_id" in added_df.columns, "Missing config_id column"
            assert set(added_df["config_id"].to_list()) == {"cfg1", "cfg2"}, (
                "Unexpected config IDs"
            )

            # Check provenance columns exist
            assert "metaxy_provenance" in added_df.columns, "Missing metaxy_provenance"
            assert "metaxy_provenance_by_field" in added_df.columns, (
                "Missing metaxy_provenance_by_field"
            )

            # Verify provenance_by_field has the expected structure
            # It should contain a hash for "combined_output" field
            for prov_by_field in added_df["metaxy_provenance_by_field"]:
                assert "combined_output" in prov_by_field, (
                    f"Expected 'combined_output' field in provenance_by_field, got: {prov_by_field}"
                )

            # === EXPAND: User code creates multiple outputs per config ===
            # Each config expands to 2 outputs, all sharing the same provenance
            expanded_rows = []
            for row in added_df.iter_rows(named=True):
                config_id = row["config_id"]
                provenance = row["metaxy_provenance"]
                provenance_by_field = row["metaxy_provenance_by_field"]

                # Create 2 outputs per config
                for output_idx in range(2):
                    expanded_rows.append(
                        {
                            "config_id": config_id,
                            "output_id": f"{config_id}_output_{output_idx}",
                            "metaxy_provenance": provenance,
                            "metaxy_provenance_by_field": provenance_by_field,
                        }
                    )

            expanded_df = pl.DataFrame(expanded_rows)

            # Write expanded downstream metadata
            any_store.write_metadata(CombinedFeature, expanded_df)

            # Verify we wrote 4 rows (2 configs × 2 outputs)
            stored_df = (
                any_store.read_metadata(CombinedFeature).lazy().collect().to_polars()
            )
            assert len(stored_df) == 4, (
                f"Expected 4 expanded rows, got {len(stored_df)}"
            )

            # === SECOND RESOLVE UPDATE (no changes) ===
            increment_no_change = any_store.resolve_update(
                CombinedFeature,
                target_version=CombinedFeature.feature_version(),
                snapshot_version=graph.snapshot_version,
            )
            added_no_change = increment_no_change.added.lazy().collect().to_polars()

            # No changes expected - upstream hasn't changed
            assert len(added_no_change) == 0, (
                f"Expected 0 rows when nothing changed, got {len(added_no_change)}"
            )

            # Snapshot the state for regression testing (at parent level for determinism)
            final_df = (
                any_store.read_metadata(CombinedFeature).lazy().collect().to_polars()
            )
            # Group by config_id to get unique provenance per config
            final_grouped = (
                final_df.group_by("config_id")
                .agg(pl.col("metaxy_provenance_by_field").first())
                .sort("config_id")
            )
            provenance_data = [
                {
                    "config_id": final_grouped["config_id"][i],
                    "metaxy_provenance_by_field": final_grouped[
                        "metaxy_provenance_by_field"
                    ][i],
                }
                for i in range(len(final_grouped))
            ]
            assert provenance_data == snapshot

    except HashAlgorithmNotSupportedError:
        pytest.skip(
            f"Hash algorithm {any_store.hash_algorithm} not supported by {any_store}"
        )


def test_expansion_changed_rows_not_duplicated(
    any_store: MetadataStore,
    graph: FeatureGraph,
):
    """Regression test: expansion lineage should not duplicate changed rows.

    When upstream changes for expansion lineage (1:N), resolve_update should return
    one "changed" row per parent, not one per expanded child. Without proper deduplication,
    the join produces N copies of each changed parent (one per child row).

    This test verifies:
    1. Initial resolve_update returns correct parent-level rows
    2. After upstream changes, "changed" contains exactly one row per parent (not duplicated)
    """

    class Video(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="video_dedup_test",
            id_columns=("video_id",),
            fields=[
                FieldSpec(key=FieldKey(["content"]), code_version="1"),
            ],
        ),
    ):
        video_id: str

    class VideoFrames(
        BaseFeature,
        spec=SampleFeatureSpec(
            key="video_frames_dedup_test",
            id_columns=("video_id", "frame_id"),
            deps=[
                FeatureDep(
                    feature=Video,
                    lineage=LineageRelationship.expansion(on=["video_id"]),
                ),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["frame_data"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                ),
            ],
        ),
    ):
        video_id: str
        frame_id: str

    try:
        with any_store:
            # === INITIAL DATA ===
            video_df = pl.DataFrame(
                {
                    "video_id": ["v1", "v2"],
                    "metaxy_provenance_by_field": [
                        {"content": "content_v1_original"},
                        {"content": "content_v2_original"},
                    ],
                }
            )
            any_store.write_metadata(Video, video_df)

            # First resolve_update - get parent-level rows
            increment = any_store.resolve_update(
                VideoFrames,
                target_version=VideoFrames.feature_version(),
                snapshot_version=graph.snapshot_version,
            )
            added_df = increment.added.lazy().collect().to_polars()
            assert len(added_df) == 2, f"Expected 2 parent rows, got {len(added_df)}"

            # Expand each video to 3 frames
            expanded_rows = []
            for row in added_df.iter_rows(named=True):
                video_id = row["video_id"]
                provenance = row["metaxy_provenance"]
                provenance_by_field = row["metaxy_provenance_by_field"]
                for frame_idx in range(3):
                    expanded_rows.append(
                        {
                            "video_id": video_id,
                            "frame_id": f"{video_id}_frame_{frame_idx}",
                            "metaxy_provenance": provenance,
                            "metaxy_provenance_by_field": provenance_by_field,
                        }
                    )

            expanded_df = pl.DataFrame(expanded_rows)
            any_store.write_metadata(VideoFrames, expanded_df)

            # Verify 6 rows stored (2 videos × 3 frames)
            stored_df = (
                any_store.read_metadata(VideoFrames).lazy().collect().to_polars()
            )
            assert len(stored_df) == 6, (
                f"Expected 6 expanded rows, got {len(stored_df)}"
            )

            # === CHANGE UPSTREAM ===
            # Update video v1's content (change provenance)
            updated_video_df = pl.DataFrame(
                {
                    "video_id": ["v1", "v2"],
                    "metaxy_provenance_by_field": [
                        {"content": "content_v1_CHANGED"},  # Changed!
                        {"content": "content_v2_original"},  # Unchanged
                    ],
                }
            )
            any_store.write_metadata(Video, updated_video_df)

            # Resolve update after upstream change
            increment_after_change = any_store.resolve_update(
                VideoFrames,
                target_version=VideoFrames.feature_version(),
                snapshot_version=graph.snapshot_version,
            )

            changed_df = increment_after_change.changed
            assert changed_df is not None, "Expected changed to not be None"
            changed_df = changed_df.lazy().collect().to_polars()

            # CRITICAL: Should be exactly 1 changed row (v1), NOT 3 (one per frame)
            # Without proper deduplication, this would return 3 rows
            assert len(changed_df) == 1, (
                f"Expected exactly 1 changed row (one per parent), got {len(changed_df)}. "
                f"This indicates expansion deduplication is broken - getting duplicate rows per child."
            )

            # Verify it's v1 that changed
            assert changed_df["video_id"].to_list() == ["v1"], (
                f"Expected v1 to be changed, got {changed_df['video_id'].to_list()}"
            )

            # Verify added is empty (no new videos)
            added_after_change = (
                increment_after_change.added.lazy().collect().to_polars()
            )
            assert len(added_after_change) == 0, (
                f"Expected 0 added rows, got {len(added_after_change)}"
            )

    except HashAlgorithmNotSupportedError:
        pytest.skip(
            f"Hash algorithm {any_store.hash_algorithm} not supported by {any_store}"
        )
