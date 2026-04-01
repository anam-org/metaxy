"""Write test pack for metadata stores."""

import random
import time
import warnings
from datetime import datetime, timedelta, timezone

import narwhals as nw
import polars as pl
import pytest
from metaxy_testing.models import SampleFeatureSpec

from metaxy import BaseFeature, FeatureDep, FeatureGraph, FeatureKey, FeatureSpec, FieldDep, FieldKey, FieldSpec
from metaxy._utils import collect_to_polars
from metaxy.metadata_store import MetadataStore
from metaxy.metadata_store.warnings import MetaxyColumnMissingWarning
from metaxy.models.constants import (
    METAXY_CREATED_AT,
    METAXY_DATA_VERSION,
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_DELETED_AT,
    METAXY_FEATURE_VERSION,
    METAXY_MATERIALIZATION_ID,
    METAXY_PROJECT_VERSION,
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
    METAXY_UPDATED_AT,
)


class WriteTests:
    """Tests for write overrides, materialization_id, and proper dataframe writes."""

    # --- Fixtures for proper dataframe write tests ---

    @pytest.fixture
    def features(self, graph: FeatureGraph) -> dict[str, type[BaseFeature]]:
        """Define test features: a root feature and a downstream feature."""

        class RootFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["root", "feature"]),
                fields=[
                    FieldSpec(key=FieldKey(["field_a"]), code_version="1"),
                    FieldSpec(key=FieldKey(["field_b"]), code_version="1"),
                ],
            ),
        ):
            """A root feature with no dependencies."""

            pass

        class DownstreamFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["downstream", "feature"]),
                deps=[
                    FeatureDep(feature=FeatureKey(["root", "feature"])),
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["output"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["root", "feature"]),
                                fields=[FieldKey(["field_a"]), FieldKey(["field_b"])],
                            )
                        ],
                    ),
                ],
            ),
        ):
            """A downstream feature that depends on the root feature."""

            pass

        return {
            "RootFeature": RootFeature,
            "DownstreamFeature": DownstreamFeature,
        }

    @pytest.fixture
    def RootFeature(self, features: dict[str, type[BaseFeature]]) -> type[BaseFeature]:
        return features["RootFeature"]

    @pytest.fixture
    def DownstreamFeature(self, features: dict[str, type[BaseFeature]]) -> type[BaseFeature]:
        return features["DownstreamFeature"]

    # --- Proper dataframe write tests ---

    def test_root_feature_with_provenance_by_field_only(
        self, store: MetadataStore, RootFeature: type[BaseFeature]
    ) -> None:
        """Root features should only require metaxy_provenance_by_field.

        When writing a root feature (no dependencies) with only
        metaxy_provenance_by_field, no warnings should be emitted since
        the provenance can be computed from provenance_by_field.
        """
        df = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"field_a": "hash_a1", "field_b": "hash_b1"},
                    {"field_a": "hash_a2", "field_b": "hash_b2"},
                    {"field_a": "hash_a3", "field_b": "hash_b3"},
                ],
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with store.open("w"):
                store.write(RootFeature, df)

                # Verify write succeeded
                result = collect_to_polars(store.read(RootFeature))
                assert len(result) == 3

            # Check for MetaxyColumnMissingWarning
            metaxy_warnings = [warning for warning in w if issubclass(warning.category, MetaxyColumnMissingWarning)]
            assert len(metaxy_warnings) == 0, (
                f"Unexpected MetaxyColumnMissingWarning: {[str(warning.message) for warning in metaxy_warnings]}"
            )

    def test_non_root_feature_with_both_provenance_columns(
        self, store: MetadataStore, RootFeature: type[BaseFeature], DownstreamFeature: type[BaseFeature]
    ) -> None:
        """Non-root features require both metaxy_provenance_by_field AND metaxy_provenance.

        When writing a non-root feature (has dependencies) with both
        provenance columns, no warnings should be emitted.
        """
        # First, write root feature data
        root_df = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"field_a": "hash_a1", "field_b": "hash_b1"},
                    {"field_a": "hash_a2", "field_b": "hash_b2"},
                    {"field_a": "hash_a3", "field_b": "hash_b3"},
                ],
            }
        )

        # Downstream feature with BOTH provenance columns
        downstream_df = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"output": "hash_out1"},
                    {"output": "hash_out2"},
                    {"output": "hash_out3"},
                ],
                METAXY_PROVENANCE: ["prov_hash1", "prov_hash2", "prov_hash3"],
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with store.open("w"):
                store.write(RootFeature, root_df)
                store.write(DownstreamFeature, downstream_df)

                # Verify write succeeded
                result = collect_to_polars(store.read(DownstreamFeature))
                assert len(result) == 3

            # Check for MetaxyColumnMissingWarning
            metaxy_warnings = [warning for warning in w if issubclass(warning.category, MetaxyColumnMissingWarning)]
            # Filter to only warnings about the downstream feature write
            # (the root feature write may also emit warnings that we'll test separately)
            assert len(metaxy_warnings) == 0, (
                f"Unexpected MetaxyColumnMissingWarning: {[str(warning.message) for warning in metaxy_warnings]}"
            )

    # --- Materialization ID tests ---

    def test_store_level_materialization_id(self, store: MetadataStore) -> None:
        """Test that store-level materialization_id is applied to all writes."""
        # Set the materialization_id on the existing store
        store._materialization_id = "test-run-123"

        key = FeatureKey(["test_mat_id"])

        class MyFeature(
            BaseFeature,
            spec=FeatureSpec(key=key, id_columns=["id"]),
        ):
            id: str

        with store.open(mode="w"):
            store.write(
                key,
                pl.DataFrame([{"id": "1", METAXY_PROVENANCE_BY_FIELD: {"default": "abc"}}]),
            )

        # Read back and verify materialization_id
        with store:
            df = store.read(key)
            assert df is not None
            df_collected = df.collect()
            assert METAXY_MATERIALIZATION_ID in df_collected.columns
            assert df_collected[METAXY_MATERIALIZATION_ID][0] == "test-run-123"

    def test_write_level_materialization_id_override(self, store: MetadataStore) -> None:
        """Test that write-level materialization_id overrides store default."""
        # Set a default materialization_id on the store
        store._materialization_id = "default-run-123"

        key = FeatureKey(["test_override"])

        class MyFeature(
            BaseFeature,
            spec=FeatureSpec(key=key, id_columns=["id"]),
        ):
            id: str

        with store.open(mode="w"):
            store.write(
                key,
                pl.DataFrame([{"id": "1", METAXY_PROVENANCE_BY_FIELD: {"default": "abc"}}]),
                materialization_id="override-run-456",
            )

        # Verify override was applied
        with store:
            df = store.read(key)
            assert df is not None
            df_collected = df.collect()
            assert df_collected[METAXY_MATERIALIZATION_ID][0] == "override-run-456"

    def test_nullable_materialization_id(self, store: MetadataStore) -> None:
        """Test that materialization_id can be null when not provided."""
        key = FeatureKey(["test_nullable"])

        class MyFeature(
            BaseFeature,
            spec=FeatureSpec(key=key, id_columns=["id"]),
        ):
            id: str

        with store.open(mode="w"):
            store.write(
                key,
                pl.DataFrame([{"id": "1", METAXY_PROVENANCE_BY_FIELD: {"default": "abc"}}]),
            )

        # Verify column exists but is null
        with store:
            df = store.read(key)
            assert df is not None
            df_collected = df.collect()
            assert METAXY_MATERIALIZATION_ID in df_collected.columns
            assert df_collected[METAXY_MATERIALIZATION_ID][0] is None

    def test_filter_by_materialization_id(self, store: MetadataStore) -> None:
        """Test filtering reads by materialization_id."""
        key = FeatureKey(["test_filter"])

        class MyFeature(
            BaseFeature,
            spec=FeatureSpec(key=key, id_columns=["id"]),
        ):
            id: str

        # Write multiple batches with different IDs
        with store.open(mode="w"):
            store.write(
                key,
                pl.DataFrame([{"id": "1", METAXY_PROVENANCE_BY_FIELD: {"default": "abc"}}]),
                materialization_id="run-1",
            )
            store.write(
                key,
                pl.DataFrame([{"id": "2", METAXY_PROVENANCE_BY_FIELD: {"default": "def"}}]),
                materialization_id="run-2",
            )
            store.write(
                key,
                pl.DataFrame([{"id": "3", METAXY_PROVENANCE_BY_FIELD: {"default": "ghi"}}]),
                materialization_id="run-1",
            )

        # Filter by run-1
        with store:
            df = store.read(key, filters=[nw.col(METAXY_MATERIALIZATION_ID) == "run-1"])
            assert df is not None
            df_collected = df.collect()
            assert len(df_collected) == 2
            assert set(df_collected["id"].to_list()) == {"1", "3"}

        # Filter by run-2
        with store:
            df = store.read(key, filters=[nw.col(METAXY_MATERIALIZATION_ID) == "run-2"])
            assert df is not None
            df_collected = df.collect()
            assert len(df_collected) == 1
            assert df_collected["id"][0] == "2"

    def test_materialization_id_multiple_writes(self, store: MetadataStore) -> None:
        """Test that different writes can have different materialization_ids."""
        key = FeatureKey(["test_multiple"])

        class MyFeature(
            BaseFeature,
            spec=FeatureSpec(key=key, id_columns=["id"]),
        ):
            id: str

        # Write with different materialization_ids
        with store.open(mode="w"):
            for i in range(3):
                store.write(
                    key,
                    pl.DataFrame(
                        [
                            {
                                "id": f"sample_{i}",
                                METAXY_PROVENANCE_BY_FIELD: {"default": f"hash_{i}"},
                            }
                        ]
                    ),
                    materialization_id=f"run-{i}",
                )

        # Read all and verify each has its materialization_id
        with store:
            df = store.read(key, with_sample_history=True)
            assert df is not None
            df_collected = df.collect()
            assert len(df_collected) == 3

            # Check that each sample has the correct materialization_id
            for i in range(3):
                sample_row = df_collected.filter(nw.col("id") == f"sample_{i}")
                assert len(sample_row) == 1
                assert sample_row[METAXY_MATERIALIZATION_ID][0] == f"run-{i}"

    # --- Write override tests ---

    def test_subsequent_writes_override_previous(self, store: MetadataStore) -> None:
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

    def test_read_returns_latest_timestamp_among_many_rows(self, store: MetadataStore) -> None:
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

        project_version = FeatureGraph.get_active().project_version

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
                    METAXY_PROJECT_VERSION: project_version,
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
            store._write_feature(store._to_table_id(key), nw.from_native(data))

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

    def test_write_overwrites_user_provided_timestamp(self, store: MetadataStore) -> None:
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

    def test_multiple_ids_each_get_latest_value(self, store: MetadataStore) -> None:
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
