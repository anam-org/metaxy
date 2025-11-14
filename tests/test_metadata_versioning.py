"""Tests for new metadata versioning functionality with metaxy_created_at and data version columns."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import narwhals as nw
import polars as pl

from metaxy import Feature, FeatureKey, FieldKey, FieldSpec, SampleFeatureSpec
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.metadata_store.memory import InMemoryMetadataStore
from metaxy.models.constants import (
    METAXY_CREATED_AT,
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_FEATURE_VERSION,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
    METAXY_SNAPSHOT_VERSION,
)
from metaxy.models.feature import FeatureGraph


class TestMetadataVersioning:
    """Test suite for metadata versioning with metaxy_created_at column."""

    def test_created_at_timestamp_added_on_write(self, tmp_path: Path) -> None:
        """Test that metaxy_created_at timestamp is automatically added when writing metadata."""
        # Create a test feature
        graph = FeatureGraph()
        with graph.use():

            class TestFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "versioning"]),
                    fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
                ),
            ):
                pass

            # Create metadata without metaxy_created_at
            df = pl.DataFrame(
                {
                    "sample_uid": ["a", "b", "c"],
                    "data": [1, 2, 3],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"data": "hash1"},
                        {"data": "hash2"},
                        {"data": "hash3"},
                    ],
                }
            )

            # Write to DuckDB store
            db_path = tmp_path / "test.duckdb"
            with DuckDBMetadataStore(db_path, auto_create_tables=True) as store:
                # Record timestamp before write
                before_write = datetime.now(timezone.utc)

                # Write metadata
                store.write_metadata(TestFeature, nw.from_native(df))

                # Record timestamp after write
                after_write = datetime.now(timezone.utc)

                # Read back metadata
                result_lazy = store.read_metadata(TestFeature)
                result_df = result_lazy.collect()

                # Convert to Polars for easier testing
                result = result_df.to_polars()

                # Verify metaxy_created_at was added
                assert METAXY_CREATED_AT in result.columns

                # Verify all rows have timestamps
                timestamps = result[METAXY_CREATED_AT].to_list()
                assert len(timestamps) == 3
                assert all(ts is not None for ts in timestamps)

                # Verify timestamps are in expected range
                for ts_str in timestamps:
                    ts = datetime.fromisoformat(ts_str)
                    assert before_write <= ts <= after_write

    def test_data_version_columns_backward_compatibility(self, tmp_path: Path) -> None:
        """Test that data version columns fall back to provenance columns when missing."""
        graph = FeatureGraph()
        with graph.use():

            class TestFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "data", "version"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

            # Create metadata with only provenance columns (no data version columns)
            df = pl.DataFrame(
                {
                    "sample_uid": ["x", "y", "z"],
                    "value": [10, 20, 30],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"value": "provenance_hash_1"},
                        {"value": "provenance_hash_2"},
                        {"value": "provenance_hash_3"},
                    ],
                    METAXY_PROVENANCE: [
                        "sample_hash_1",
                        "sample_hash_2",
                        "sample_hash_3",
                    ],
                }
            )

            # Write to store
            db_path = tmp_path / "test_backward_compat.duckdb"
            with DuckDBMetadataStore(db_path, auto_create_tables=True) as store:
                store.write_metadata(TestFeature, nw.from_native(df))

                # Read back metadata
                result = store.read_metadata(TestFeature).collect().to_polars()

                # Verify data version columns were added with values from provenance
                assert METAXY_DATA_VERSION_BY_FIELD in result.columns
                assert METAXY_DATA_VERSION in result.columns

                # Verify data version columns match provenance columns
                for i in range(len(result)):
                    assert (
                        result[METAXY_DATA_VERSION_BY_FIELD][i]
                        == result[METAXY_PROVENANCE_BY_FIELD][i]
                    )
                    assert (
                        result[METAXY_DATA_VERSION][i] == result[METAXY_PROVENANCE][i]
                    )

    def test_latest_record_selection_with_created_at(self, tmp_path: Path) -> None:
        """Test that only the latest records are used when multiple versions exist."""
        graph = FeatureGraph()
        with graph.use():

            class TestFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "latest"]),
                    fields=[FieldSpec(key=FieldKey(["content"]), code_version="1")],
                ),
            ):
                pass

            db_path = tmp_path / "test_latest.duckdb"
            with DuckDBMetadataStore(db_path, auto_create_tables=True) as store:
                # Write initial metadata
                df1 = pl.DataFrame(
                    {
                        "sample_uid": ["a", "b"],
                        "content": ["old_a", "old_b"],
                        METAXY_PROVENANCE_BY_FIELD: [
                            {"content": "hash_old_a"},
                            {"content": "hash_old_b"},
                        ],
                    }
                )
                store.write_metadata(TestFeature, nw.from_native(df1))

                # Sleep briefly to ensure different timestamps
                time.sleep(0.1)

                # Write updated metadata for sample 'a' with newer content
                # This simulates manual provenance value changes
                df2 = pl.DataFrame(
                    {
                        "sample_uid": ["a"],
                        "content": ["new_a"],
                        METAXY_PROVENANCE_BY_FIELD: [{"content": "hash_new_a"}],
                        METAXY_FEATURE_VERSION: [TestFeature.feature_version()],
                        METAXY_SNAPSHOT_VERSION: [graph.snapshot_version],
                    }
                )
                store.write_metadata(TestFeature, nw.from_native(df2))

                # Now when we compute increments, it should use the latest record for 'a'
                # This tests the logic in resolve_incremental_update
                increment = store.resolve_update(TestFeature, lazy=False)

                # The increment should be empty because we're comparing latest vs latest
                assert isinstance(increment.added, pl.DataFrame)
                assert len(increment.added) == 0  # No new samples

                # Read all metadata (without filtering by latest)
                all_records = (
                    store.read_metadata(TestFeature, current_only=False)
                    .collect()
                    .to_polars()
                )

                # We should have 3 records total (2 from first write, 1 from second)
                assert len(all_records) == 3

    def test_multiple_versions_same_sample(self, tmp_path: Path) -> None:
        """Test handling multiple versions of the same sample with different timestamps."""
        graph = FeatureGraph()
        with graph.use():

            class VersionedFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["versioned", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
                ),
            ):
                pass

            # Use InMemoryMetadataStore for simpler testing
            with InMemoryMetadataStore() as store:
                # Write version 1
                timestamp1 = "2024-01-01T10:00:00+00:00"
                df1 = pl.DataFrame(
                    {
                        "sample_uid": ["sample1", "sample2"],
                        "data": ["v1_data1", "v1_data2"],
                        METAXY_PROVENANCE_BY_FIELD: [
                            {"data": "v1_hash1"},
                            {"data": "v1_hash2"},
                        ],
                        METAXY_CREATED_AT: [timestamp1, timestamp1],
                        METAXY_FEATURE_VERSION: [VersionedFeature.feature_version()]
                        * 2,
                        METAXY_SNAPSHOT_VERSION: [graph.snapshot_version] * 2,
                    }
                )
                store.write_metadata(VersionedFeature, nw.from_native(df1))

                # Write version 2 with newer timestamp for sample1 only
                timestamp2 = "2024-01-01T11:00:00+00:00"
                df2 = pl.DataFrame(
                    {
                        "sample_uid": ["sample1"],
                        "data": ["v2_data1"],
                        METAXY_PROVENANCE_BY_FIELD: [{"data": "v2_hash1"}],
                        METAXY_CREATED_AT: [timestamp2],
                        METAXY_FEATURE_VERSION: [VersionedFeature.feature_version()],
                        METAXY_SNAPSHOT_VERSION: [graph.snapshot_version],
                    }
                )
                store.write_metadata(VersionedFeature, nw.from_native(df2))

                # Read all records
                all_records = (
                    store.read_metadata(VersionedFeature, current_only=False)
                    .collect()
                    .to_polars()
                )

                # Should have 3 records total
                assert len(all_records) == 3

                # Check timestamps are preserved
                timestamps = sorted(all_records[METAXY_CREATED_AT].to_list())
                assert timestamps[0] == timestamp1
                assert timestamps[1] == timestamp1
                assert timestamps[2] == timestamp2

    def test_data_version_columns_explicit_values(self, tmp_path: Path) -> None:
        """Test that explicitly provided data version columns are preserved."""
        graph = FeatureGraph()
        with graph.use():

            class DataVersionFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["explicit", "data", "version"]),
                    fields=[
                        FieldSpec(key=FieldKey(["field1"]), code_version="1"),
                        FieldSpec(key=FieldKey(["field2"]), code_version="1"),
                    ],
                ),
            ):
                pass

            # Create metadata with explicit data version columns (different from provenance)
            df = pl.DataFrame(
                {
                    "sample_uid": ["a", "b"],
                    "field1": [1, 2],
                    "field2": [3, 4],
                    METAXY_PROVENANCE_BY_FIELD: [
                        {"field1": "prov_f1_a", "field2": "prov_f2_a"},
                        {"field1": "prov_f1_b", "field2": "prov_f2_b"},
                    ],
                    METAXY_PROVENANCE: ["prov_sample_a", "prov_sample_b"],
                    # Explicit data versions (different from provenance)
                    METAXY_DATA_VERSION_BY_FIELD: [
                        {"field1": "data_f1_a", "field2": "data_f2_a"},
                        {"field1": "data_f1_b", "field2": "data_f2_b"},
                    ],
                    METAXY_DATA_VERSION: ["data_sample_a", "data_sample_b"],
                }
            )

            db_path = tmp_path / "test_explicit.duckdb"
            with DuckDBMetadataStore(db_path, auto_create_tables=True) as store:
                store.write_metadata(DataVersionFeature, nw.from_native(df))

                # Read back metadata
                result = store.read_metadata(DataVersionFeature).collect().to_polars()

                # Verify explicit data version values were preserved
                assert result[METAXY_DATA_VERSION][0] == "data_sample_a"
                assert result[METAXY_DATA_VERSION][1] == "data_sample_b"

                # Verify data version is different from provenance
                assert result[METAXY_DATA_VERSION][0] != result[METAXY_PROVENANCE][0]
                assert result[METAXY_DATA_VERSION][1] != result[METAXY_PROVENANCE][1]

    def test_sqlmodel_integration_new_columns(self) -> None:
        """Test that SQLModel integration includes the new columns."""
        from metaxy.ext.sqlmodel import BaseSQLModelFeature

        # Verify new fields exist in BaseSQLModelFeature
        assert hasattr(BaseSQLModelFeature, "metaxy_created_at")
        assert hasattr(BaseSQLModelFeature, "metaxy_data_version_by_field")
        assert hasattr(BaseSQLModelFeature, "metaxy_data_version")

        # Create a concrete SQLModel feature
        graph = FeatureGraph()
        with graph.use():

            class SQLModelTestFeature(
                BaseSQLModelFeature,
                table=True,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["sqlmodel", "test"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                __tablename__: str = "sqlmodel_test"  # pyright: ignore[reportIncompatibleVariableOverride]
                sample_uid: str

            # Verify the class has all system columns
            instance = SQLModelTestFeature(sample_uid="test")
            assert hasattr(instance, "metaxy_created_at")
            assert hasattr(instance, "metaxy_data_version")
            assert hasattr(instance, "metaxy_data_version_by_field")
