"""Unit tests for latest record selection using metaxy_created_at."""

from datetime import datetime, timezone
from pathlib import Path

import narwhals as nw
import polars as pl

from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, SampleFeatureSpec
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.models.feature import FeatureGraph


def test_rank_over_descending_selects_latest_record(tmp_path: Path):
    """Test that rank().over(descending=True) correctly selects the latest record."""
    # Create test data with multiple versions
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b", "c"],
            "value": ["old_a", "new_a", "old_b", "new_b", "only_c"],
            "created_at": [
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 2, tzinfo=timezone.utc),  # Latest for 'a'
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 3, tzinfo=timezone.utc),  # Latest for 'b'
                datetime(2024, 1, 1, tzinfo=timezone.utc),
            ],
        }
    )

    df_nw = nw.from_native(df.lazy())
    id_columns = ["id"]

    # Apply the rank filter logic
    result = (
        df_nw.with_columns(
            nw.col("created_at")
            .rank(method="ordinal", descending=True)
            .over(id_columns)
            .alias("__row_num")
        )
        .filter(nw.col("__row_num") == 1)
        .drop("__row_num")
        .collect()
        .to_native()
    )

    # Should have 3 records (latest for each ID)
    assert len(result) == 3

    # Check we got the latest values
    assert result.filter(pl.col("id") == "a")["value"][0] == "new_a"
    assert result.filter(pl.col("id") == "b")["value"][0] == "new_b"
    assert result.filter(pl.col("id") == "c")["value"][0] == "only_c"

    # Verify timestamps are the latest
    a_time = result.filter(pl.col("id") == "a")["created_at"][0]
    assert a_time == datetime(2024, 1, 2, tzinfo=timezone.utc)

    b_time = result.filter(pl.col("id") == "b")["created_at"][0]
    assert b_time == datetime(2024, 1, 3, tzinfo=timezone.utc)


def test_metaxy_created_at_deduplication_in_resolve_update(tmp_path: Path):
    """Test that resolve_update correctly uses latest records based on created_at."""
    graph = FeatureGraph()
    with graph.use():

        class TestFeature(
            Feature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "dedup"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

        db_path = tmp_path / "test_dedup.duckdb"
        with DuckDBMetadataStore(db_path, auto_create_tables=True) as store:
            # Write first version
            df1 = pl.DataFrame(
                {
                    "sample_uid": ["s1", "s2"],
                    "data": ["v1_s1", "v1_s2"],
                    "metaxy_provenance_by_field": [
                        {"data": "hash1"},
                        {"data": "hash2"},
                    ],
                }
            )
            store.write_metadata(TestFeature, df1)

            # Get the first timestamp
            first_records = (
                store.read_metadata(TestFeature, current_only=False)
                .collect()
                .to_polars()
            )
            first_records["metaxy_created_at"][0]

            # Write second version for s1 only with newer timestamp
            import time

            time.sleep(0.1)  # Ensure different timestamp

            df2 = pl.DataFrame(
                {
                    "sample_uid": ["s1"],
                    "data": ["v2_s1"],
                    "metaxy_provenance_by_field": [
                        {"data": "hash1_v2"},
                    ],
                    "metaxy_feature_version": [TestFeature.feature_version()],
                    "metaxy_snapshot_version": [graph.snapshot_version],
                }
            )
            store.write_metadata(TestFeature, df2)

            # Read all records - should have 3 total (s1 v1, s1 v2, s2 v1)
            all_records = (
                store.read_metadata(TestFeature, current_only=False)
                .collect()
                .to_polars()
            )
            assert len(all_records) == 3

            # Read current only - should have 2 records (latest s1, s2)
            current_records = (
                store.read_metadata(TestFeature, current_only=True)
                .collect()
                .to_polars()
            )
            assert len(current_records) == 2

            # Verify s1 has the latest version
            s1_record = current_records.filter(pl.col("sample_uid") == "s1")
            assert len(s1_record) == 1
            assert s1_record["data"][0] == "v2_s1"
            assert s1_record["metaxy_provenance_by_field"][0] == {"data": "hash1_v2"}

            # Verify s1's timestamp is newer than s2's
            s1_time = s1_record["metaxy_created_at"][0]
            s2_time = current_records.filter(pl.col("sample_uid") == "s2")[
                "metaxy_created_at"
            ][0]
            assert s1_time > s2_time
