"""Tests verifying that subsequent writes override previous values."""

import random
import time
from datetime import datetime, timedelta, timezone

import narwhals as nw
import polars as pl
from pytest_cases import parametrize_with_cases

from metaxy import BaseFeature, FeatureKey, FeatureSpec
from metaxy.metadata_store import MetadataStore
from metaxy.models.constants import (
    METAXY_CREATED_AT,
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_DELETED_AT,
    METAXY_FEATURE_VERSION,
    METAXY_MATERIALIZATION_ID,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
    METAXY_SNAPSHOT_VERSION,
    METAXY_UPDATED_AT,
)

from .conftest import AllStoresCases


@parametrize_with_cases("store", cases=AllStoresCases)
def test_subsequent_writes_override_previous(store: MetadataStore):
    """Test that subsequent writes override previous values without explicit action.

    Performs 5 writes to the same row, incrementing a value column each time.
    Verifies that read returns only the last written value.
    """
    key = FeatureKey(["test_write_override"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str
        value: int

    with store.open("w"):
        # Perform 5 writes with incrementing values
        for i in range(1, 6):
            data = pl.DataFrame(
                {
                    "id": ["row1"],
                    "value": [i * 10],  # 10, 20, 30, 40, 50
                    METAXY_PROVENANCE_BY_FIELD: [{"default": f"hash_v{i}"}],
                }
            )
            store.write(key, data)
            # Small delay to ensure different timestamps
            time.sleep(0.01)

        # Read with deduplication (default behavior)
        result = store.read(key, with_sample_history=False).collect().to_polars()

        # Should get exactly 1 row
        assert result.shape[0] == 1, f"Expected 1 row, got {result.shape[0]}"

        # Should have the last written value (50)
        assert result["value"][0] == 50, f"Expected value=50, got {result['value'][0]}"

        # Verify all 5 versions exist when reading without deduplication
        all_versions = store.read(key, with_sample_history=True).collect().to_polars()
        assert all_versions.shape[0] == 5, f"Expected 5 versions, got {all_versions.shape[0]}"


@parametrize_with_cases("store", cases=AllStoresCases)
def test_read_returns_latest_timestamp_among_many_rows(store: MetadataStore):
    """Test that read returns the row with the latest timestamp.

    Writes 100 rows with controlled metaxy_updated_at timestamps using
    _write_feature (bypassing timestamp auto-generation). The row with
    the latest timestamp should be returned by read with with_sample_history=False.
    """
    key = FeatureKey(["test_latest_timestamp"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str
        value: int

    num_rows = 100
    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    # The row with value=42 will have the LATEST timestamp
    latest_value = 42
    latest_time = base_time + timedelta(days=365)  # 1 year later than all others

    # Get version info from the feature class
    feature_version = MyFeature.feature_version()

    # First, write a single row using write to create the table with proper schema
    with store.open("w"):
        init_data = pl.DataFrame(
            {
                "id": ["init"],
                "value": [-1],
                METAXY_PROVENANCE_BY_FIELD: [{"default": "init_hash"}],
            }
        )
        store.write(key, init_data)

    # Get snapshot version from the graph after first write
    from metaxy.models.feature import FeatureGraph

    snapshot_version = FeatureGraph.get_active().snapshot_version

    # Now create 100 rows with controlled timestamps
    rows = []
    for i in range(num_rows):
        if i == latest_value:
            ts = latest_time
        else:
            ts = base_time + timedelta(seconds=i)

        rows.append(
            {
                "id": "same_id",
                "value": i,
                # All required system columns for _write_feature
                METAXY_PROVENANCE_BY_FIELD: {"default": f"hash_{i}"},
                METAXY_PROVENANCE: f"provenance_{i}",
                METAXY_DATA_VERSION_BY_FIELD: {"default": f"hash_{i}"},
                METAXY_DATA_VERSION: f"data_version_{i}",
                METAXY_FEATURE_VERSION: feature_version,
                METAXY_SNAPSHOT_VERSION: snapshot_version,
                METAXY_CREATED_AT: ts,
                METAXY_UPDATED_AT: ts,
                METAXY_DELETED_AT: None,
                METAXY_MATERIALIZATION_ID: None,
            }
        )

    # Shuffle to ensure order doesn't matter
    random.shuffle(rows)

    # Create DataFrame with proper types for nullable columns
    data = pl.DataFrame(rows).cast(
        {
            METAXY_DELETED_AT: pl.Datetime(time_zone="UTC"),
            METAXY_MATERIALIZATION_ID: pl.String,
        }
    )

    # Use _write_feature to bypass timestamp auto-generation
    with store.open("w"):
        store._write_feature(key, nw.from_native(data))

    # Read and verify
    with store:
        # Read all rows for "same_id" (excluding our init row)
        all_rows = store.read(key, with_sample_history=True).collect().to_polars().filter(pl.col("id") == "same_id")
        assert all_rows.shape[0] == num_rows, f"Expected {num_rows} rows, got {all_rows.shape[0]}"

        # Verify timestamps are what we set (not auto-generated)
        row_42 = all_rows.filter(pl.col("value") == latest_value)
        assert row_42.shape[0] == 1
        actual_ts = row_42[METAXY_UPDATED_AT][0]
        assert actual_ts == latest_time, (
            f"Expected timestamp {latest_time} for value=42, got {actual_ts}. Timestamp was likely overwritten."
        )

        # Verify that the max timestamp in our data is indeed latest_time
        max_ts = all_rows[METAXY_UPDATED_AT].max()
        assert max_ts == latest_time, f"Expected max timestamp {latest_time}, got {max_ts}"

        # Read with deduplication - should get only the row with latest timestamp
        result = store.read(key, with_sample_history=False).collect().to_polars().filter(pl.col("id") == "same_id")

        # Should get exactly 1 row
        assert result.shape[0] == 1, f"Expected 1 row, got {result.shape[0]}"

        # Should have value=42 (the row with the latest timestamp, NOT value=99)
        assert result["value"][0] == latest_value, (
            f"Expected value={latest_value} (row with latest timestamp), "
            f"got {result['value'][0]}. Deduplication may not be using metaxy_updated_at correctly."
        )

        # Verify the returned row has the expected timestamp (which is the max)
        assert result[METAXY_UPDATED_AT][0] == latest_time
        assert result[METAXY_UPDATED_AT][0] == max_ts


@parametrize_with_cases("store", cases=AllStoresCases)
def test_write_overwrites_user_provided_timestamp(store: MetadataStore):
    """Verify that write overwrites user-provided metaxy_updated_at.

    Users should not be able to set arbitrary timestamps via write.
    The system always sets metaxy_updated_at to the current time on write.
    To preserve custom timestamps, use _write_feature directly.
    """
    key = FeatureKey(["test_timestamp_overwrite"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str
        value: int

    # Create a timestamp far in the past
    past_time = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Write data WITH metaxy_updated_at already set to past_time
    data = pl.DataFrame(
        {
            "id": ["row1"],
            "value": [100],
            METAXY_UPDATED_AT: [past_time],
            METAXY_PROVENANCE_BY_FIELD: [{"default": "hash1"}],
        }
    )

    with store.open("w"):
        store.write(key, data)

    with store:
        result = store.read(key, with_sample_history=True).collect().to_polars()
        actual_ts = result[METAXY_UPDATED_AT][0]

        # The timestamp should be overwritten to current time, NOT preserved
        assert actual_ts > past_time, (
            f"Expected metaxy_updated_at to be overwritten to current time, "
            f"but got {actual_ts} which is not after {past_time}"
        )

        # Verify it's close to "now" (within last minute)
        now = datetime.now(timezone.utc)
        time_diff = now - actual_ts
        assert time_diff.total_seconds() < 60, (
            f"Expected metaxy_updated_at to be recent (within 60s of now), "
            f"but got {actual_ts} which is {time_diff.total_seconds()}s ago"
        )


@parametrize_with_cases("store", cases=AllStoresCases)
def test_multiple_ids_each_get_latest_value(store: MetadataStore):
    """Test that multiple different IDs each get their latest value independently.

    Writes multiple versions for multiple IDs and verifies each ID gets its own latest.
    """
    key = FeatureKey(["test_multi_id_override"])

    class MyFeature(
        BaseFeature,
        spec=FeatureSpec(key=key, id_columns=["id"]),
    ):
        id: str
        value: int

    with store.open("w"):
        # Write 3 versions for each of 3 IDs
        for version in range(1, 4):
            data = pl.DataFrame(
                {
                    "id": ["a", "b", "c"],
                    "value": [version * 10, version * 100, version * 1000],
                    METAXY_PROVENANCE_BY_FIELD: [{"default": f"hash_{id_}_v{version}"} for id_ in ["a", "b", "c"]],
                }
            )
            store.write(key, data)
            time.sleep(0.01)

        # Read with deduplication
        result = store.read(key, with_sample_history=False).collect().to_polars()

        # Should get 3 rows (one per ID)
        assert result.shape[0] == 3, f"Expected 3 rows, got {result.shape[0]}"

        # Convert to dict for easier checking
        result_dict = {row["id"]: row["value"] for row in result.iter_rows(named=True)}

        # Each ID should have its latest value (version 3)
        assert result_dict["a"] == 30, f"Expected a=30, got {result_dict['a']}"
        assert result_dict["b"] == 300, f"Expected b=300, got {result_dict['b']}"
        assert result_dict["c"] == 3000, f"Expected c=3000, got {result_dict['c']}"

        # Verify all 9 versions exist when reading without deduplication
        all_versions = store.read(key, with_sample_history=True).collect().to_polars()
        assert all_versions.shape[0] == 9, f"Expected 9 versions, got {all_versions.shape[0]}"
