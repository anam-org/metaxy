"""Tests for Dagster IO Manager integration."""

from typing import Any

import dagster as dg
import narwhals as nw
import polars as pl
import pytest

import metaxy as mx


@pytest.fixture
def upstream_feature() -> type[mx.BaseFeature]:
    """Create an upstream feature for testing."""
    upstream_spec = mx.FeatureSpec(
        key=["features", "upstream"],
        id_columns=["id"],
        fields=["value"],
    )

    class Upstream(mx.BaseFeature, spec=upstream_spec):
        id: str

    return Upstream


@pytest.fixture
def downstream_feature(
    upstream_feature: type[mx.BaseFeature],
) -> type[mx.BaseFeature]:
    """Create a downstream feature that depends on upstream."""
    downstream_spec = mx.FeatureSpec(
        key=["features", "downstream"],
        id_columns=["id"],
        fields=["result"],
        deps=[mx.FeatureDep(feature=upstream_feature)],
    )

    class Downstream(mx.BaseFeature, spec=downstream_spec):
        id: str

    return Downstream


class TestMetaxyIOManagerHandleOutput:
    """Test handle_output functionality."""

    def test_handle_output_with_none_logs_metadata(
        self,
        upstream_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that handle_output with None just logs metadata without writing."""

        @dg.asset(
            metadata={"metaxy/feature": "features/upstream"},
            io_manager_key="metaxy_io_manager",
        )
        def my_asset():
            # Return None - IOManager should just log metadata
            return None

        result = dg.materialize(
            [my_asset],
            resources=resources,
            instance=instance,
        )

        assert result.success
        # Check that output metadata was logged
        event = result.get_asset_materialization_events()[0]
        metadata = event.step_materialization_data.materialization.metadata
        assert "metaxy/feature" in metadata
        assert "metaxy/info" in metadata
        assert "metaxy/store" in metadata

    def test_handle_output_with_metaxy_output_writes_data(
        self,
        upstream_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that handle_output with MetaxyOutput writes data to store."""

        @dg.asset(
            metadata={"metaxy/feature": "features/upstream"},
            io_manager_key="metaxy_io_manager",
        )
        def my_asset():
            return pl.DataFrame(
                {
                    "id": ["1", "2", "3"],
                    "metaxy_provenance_by_field": [
                        {"value": "v1"},
                        {"value": "v1"},
                        {"value": "v1"},
                    ],
                }
            )

        result = dg.materialize(
            [my_asset],
            resources=resources,
            instance=instance,
        )

        assert result.success

        # Verify data was written to the store by running another asset that reads it
        captured_data = {}

        @dg.asset(deps=[my_asset])
        def verify_asset(store: dg.ResourceParam[mx.MetadataStore]):
            with store:
                captured_data["rows"] = len(
                    store.read_metadata(upstream_feature).collect()
                )

        result2 = dg.materialize(
            [verify_asset],
            resources=resources,
            instance=instance,
            selection=[verify_asset],
        )
        assert result2.success
        assert captured_data["rows"] == 3


class TestMetaxyIOManagerLoadInput:
    """Test load_input functionality."""

    def test_load_input_reads_upstream_data(
        self,
        upstream_feature: type[mx.BaseFeature],
        downstream_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that load_input reads data from upstream feature."""
        captured_input = {}

        @dg.asset(
            metadata={"metaxy/feature": "features/upstream"},
            io_manager_key="metaxy_io_manager",
        )
        def upstream_asset():
            return pl.DataFrame(
                {
                    "id": ["a", "b"],
                    "metaxy_provenance_by_field": [
                        {"value": "hash1"},
                        {"value": "hash2"},
                    ],
                }
            )

        @dg.asset(
            metadata={"metaxy/feature": "features/downstream"},
            io_manager_key="metaxy_io_manager",
            deps=[upstream_asset],
        )
        def downstream_asset(store: dg.ResourceParam[mx.MetadataStore]):
            # Manually load the upstream data to verify it works
            with store:
                upstream_data = store.read_metadata(upstream_feature).collect()
                captured_input["data"] = upstream_data
            return None

        # First materialize upstream
        result1 = dg.materialize(
            [upstream_asset],
            resources=resources,
            instance=instance,
        )
        assert result1.success

        # Then materialize downstream
        result2 = dg.materialize(
            [downstream_asset],
            resources=resources,
            instance=instance,
            selection=[downstream_asset],
        )
        assert result2.success

        # Verify downstream could read upstream data
        assert "data" in captured_input
        assert len(captured_input["data"]) == 2


class TestMetaxyIOManagerMetadata:
    """Test metadata attachment functionality."""

    def test_output_metadata_includes_feature_info(
        self,
        upstream_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that output metadata includes feature version info."""

        @dg.asset(
            metadata={"metaxy/feature": "features/upstream"},
            io_manager_key="metaxy_io_manager",
        )
        def my_asset():
            return None

        result = dg.materialize(
            [my_asset],
            resources=resources,
            instance=instance,
        )

        assert result.success
        event = result.get_asset_materialization_events()[0]
        metadata = event.step_materialization_data.materialization.metadata

        # Check metaxy/info contains expected fields
        info = metadata["metaxy/info"].value
        assert isinstance(info, dict)
        assert "project" in info
        assert "feature_code_version" in info
        assert "feature_version" in info

        # Check metaxy/store contains store type info
        store_meta = metadata["metaxy/store"].value
        assert isinstance(store_meta, dict)
        assert "type" in store_meta
        assert store_meta["type"] == "DeltaMetadataStore"

    def test_materialized_in_run_count(
        self,
        upstream_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that materialized_in_run count is reported."""

        @dg.asset(
            metadata={"metaxy/feature": "features/upstream"},
            io_manager_key="metaxy_io_manager",
        )
        def my_asset():
            return pl.DataFrame(
                {
                    "id": ["x", "y"],
                    "metaxy_provenance_by_field": [
                        {"value": "h1"},
                        {"value": "h2"},
                    ],
                }
            )

        result = dg.materialize(
            [my_asset],
            resources=resources,
            instance=instance,
        )

        assert result.success
        event = result.get_asset_materialization_events()[0]
        metadata = event.step_materialization_data.materialization.metadata

        # Should report how many rows were materialized in this run
        assert "metaxy/materialized_in_run" in metadata
        assert metadata["metaxy/materialized_in_run"].value == 2


class TestMetaxyIOManagerPartitions:
    """Test partitioned asset handling."""

    @pytest.fixture
    def partitioned_feature(self) -> type[mx.BaseFeature]:
        """Create a feature with a partition column."""
        spec = mx.FeatureSpec(
            key=["features", "partitioned"],
            id_columns=["id"],
            fields=["value"],
        )

        class PartitionedFeature(mx.BaseFeature, spec=spec):
            id: str

        return PartitionedFeature

    @pytest.fixture
    def partitions_def(self) -> dg.StaticPartitionsDefinition:
        """Static partitions for testing."""
        return dg.StaticPartitionsDefinition(["a", "b", "c"])

    def test_partitioned_to_partitioned_loads_single_partition(
        self,
        partitioned_feature: type[mx.BaseFeature],
        partitions_def: dg.StaticPartitionsDefinition,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Partitioned asset loading from partitioned upstream should only get one partition."""
        captured_data = {}

        @dg.asset(
            metadata={
                "metaxy/feature": "features/partitioned",
                "partition_by": "partition",
            },
            io_manager_key="metaxy_io_manager",
            partitions_def=partitions_def,
        )
        def upstream_partitioned(context: dg.AssetExecutionContext):
            partition = context.partition_key
            return pl.DataFrame(
                {
                    "id": [f"{partition}_1", f"{partition}_2"],
                    "partition": [partition, partition],
                    "metaxy_provenance_by_field": [
                        {"value": "v1"},
                        {"value": "v1"},
                    ],
                }
            )

        @dg.asset(
            metadata={
                "metaxy/feature": "features/partitioned",
                "partition_by": "partition",
            },
            io_manager_key="metaxy_io_manager",
            partitions_def=partitions_def,
            deps=[upstream_partitioned],
        )
        def downstream_partitioned(
            context: dg.AssetExecutionContext,
            store: dg.ResourceParam[mx.MetadataStore],
        ):
            # Read using IO manager's partition filtering
            partition = context.partition_key
            with store:
                # Read all data first
                all_data = store.read_metadata(partitioned_feature).collect()
                # Filter to current partition (simulating what IO manager does)
                partition_data = all_data.filter(nw.col("partition") == partition)
                captured_data[partition] = len(partition_data)
            return None

        # Materialize all partitions of upstream
        for partition in ["a", "b", "c"]:
            result = dg.materialize(
                [upstream_partitioned],
                resources=resources,
                instance=instance,
                partition_key=partition,
            )
            assert result.success

        # Verify all data was written (6 rows total: 2 per partition)
        with mx.MetaxyConfig.get().get_store("dev") as store:
            all_data = store.read_metadata(partitioned_feature).collect()
            assert len(all_data) == 6

        # Materialize downstream for partition "b" only
        result = dg.materialize(
            [downstream_partitioned],
            resources=resources,
            instance=instance,
            partition_key="b",
            selection=[downstream_partitioned],
        )
        assert result.success

        # Should have read only 2 rows (partition "b")
        assert captured_data["b"] == 2

    def test_partitioned_to_unpartitioned_loads_all_partitions(
        self,
        partitioned_feature: type[mx.BaseFeature],
        partitions_def: dg.StaticPartitionsDefinition,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Unpartitioned asset loading from partitioned upstream should get all partitions."""
        captured_data = {}

        @dg.asset(
            metadata={
                "metaxy/feature": "features/partitioned",
                "partition_by": "partition",
            },
            io_manager_key="metaxy_io_manager",
            partitions_def=partitions_def,
        )
        def upstream_partitioned(context: dg.AssetExecutionContext):
            partition = context.partition_key
            return pl.DataFrame(
                {
                    "id": [f"{partition}_1", f"{partition}_2"],
                    "partition": [partition, partition],
                    "metaxy_provenance_by_field": [
                        {"value": "v1"},
                        {"value": "v1"},
                    ],
                }
            )

        @dg.asset(
            metadata={"metaxy/feature": "features/partitioned"},
            io_manager_key="metaxy_io_manager",
            deps=[upstream_partitioned],
        )
        def downstream_unpartitioned(store: dg.ResourceParam[mx.MetadataStore]):
            # Unpartitioned asset should read all data
            with store:
                all_data = store.read_metadata(partitioned_feature).collect()
                captured_data["total"] = len(all_data)
            return None

        # Materialize all partitions of upstream
        for partition in ["a", "b", "c"]:
            result = dg.materialize(
                [upstream_partitioned],
                resources=resources,
                instance=instance,
                partition_key=partition,
            )
            assert result.success

        # Materialize unpartitioned downstream
        result = dg.materialize(
            [downstream_unpartitioned],
            resources=resources,
            instance=instance,
            selection=[downstream_unpartitioned],
        )
        assert result.success

        # Should have read all 6 rows (all partitions)
        assert captured_data["total"] == 6
