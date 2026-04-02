"""Unit tests for keep_latest_by_group across versioning engines.

These tests verify the deduplication utility independently of any metadata store.
They are parametrized across Polars and Ibis versioning engines.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest_cases
from pytest_cases import parametrize_with_cases

from metaxy.utils import collect_to_polars


class KeepLatestTestDataCases:
    """Test data cases for keep_latest_by_group tests."""

    def case_polars(self) -> tuple[Any, Any, datetime]:
        import narwhals as nw

        from metaxy.versioning.polars import PolarsVersioningEngine

        base_time = datetime(2024, 1, 1, 12, 0, 0)

        def create_data_fn(pl_df):
            return nw.from_native(pl_df)

        return (PolarsVersioningEngine, create_data_fn, base_time)

    def case_ibis(self, tmp_path: Path) -> tuple[Any, Any, datetime]:
        import ibis
        import narwhals as nw

        from metaxy.ext.ibis.versioning import IbisVersioningEngine

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
def keep_latest_test_data(test_data: tuple[Any, Any, datetime]) -> tuple[Any, Any, datetime]:
    return test_data


def test_keep_latest_by_group(keep_latest_test_data: tuple[Any, Any, datetime]) -> None:
    from datetime import timedelta

    import polars as pl

    # Get fixture data
    engine_class, create_data_fn, base_time = keep_latest_test_data

    # Create test data with 5 versions of the same sample
    data = pl.DataFrame(
        {
            "sample_uid": ["sample1"] * 5,
            "value": [10, 20, 30, 40, 50],  # Different values per version
            "timestamp": [base_time + timedelta(hours=i) for i in range(5)],  # Increasing timestamps
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
        timestamp_columns=["timestamp"],
    )

    # Convert result to Polars for assertion
    result = collect_to_polars(result_nw)

    # Verify we got exactly 1 row (only the latest)
    assert len(result) == 1, f"Expected 1 row, got {len(result)}"

    # Verify it's the latest version (value=50)
    assert result["value"][0] == 50, f"Expected value=50 (latest), got {result['value'][0]}"

    # Verify the timestamp is the latest
    expected_timestamp = base_time + timedelta(hours=4)
    assert result["timestamp"][0] == expected_timestamp, (
        f"Expected timestamp={expected_timestamp}, got {result['timestamp'][0]}"
    )


def test_keep_latest_by_group_aggregation_n_to_1(keep_latest_test_data: tuple[Any, Any, datetime]) -> None:
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
            "timestamp": [base_time + timedelta(hours=i) for i in range(8)],  # Increasing timestamps
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
        timestamp_columns=["timestamp"],
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


def test_keep_latest_by_group_expansion_1_to_n(keep_latest_test_data: tuple[Any, Any, datetime]) -> None:
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
            "timestamp": [base_time + timedelta(hours=i) for i in range(5)],  # Increasing timestamps
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
        timestamp_columns=["timestamp"],
    )

    # Convert result to Polars for assertion
    result = collect_to_polars(result_nw)

    # Verify we got exactly 2 rows (one per video_id: v1, v2)
    assert len(result) == 2, f"Expected 2 rows (one per video), got {len(result)}"

    # Verify only latest versions kept
    result_sorted = result.sort("video_id")

    # v1's latest version is "4K" (3rd occurrence, timestamp +2 hours)
    # v2's latest version is "1080p" (2nd occurrence, timestamp +4 hours)
    assert result_sorted["resolution"].to_list() == ["4K", "1080p"], "Expected latest versions: v1=4K, v2=1080p"
    assert result_sorted["fps"].to_list() == [60, 60], "Expected latest fps values"
