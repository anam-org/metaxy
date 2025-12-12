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

    def test_handle_output_with_none_logs_metadata_when_data_exists(
        self,
        upstream_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that handle_output with None logs metadata when feature data exists externally."""
        # First write data externally (not via IOManager)
        store = mx.MetaxyConfig.get().get_store("dev")
        with store.open("write"):
            store.write_metadata(
                upstream_feature,
                pl.DataFrame(
                    {
                        "id": ["ext1", "ext2"],
                        "metaxy_provenance_by_field": [
                            {"value": "v1"},
                            {"value": "v2"},
                        ],
                    }
                ),
            )

        @dg.asset(
            metadata={"metaxy/feature": "features/upstream"},
            io_manager_key="metaxy_io_manager",
        )
        def my_asset():
            # Return None - data was written externally, IOManager should still log metadata
            return None

        result = dg.materialize(
            [my_asset],
            resources=resources,
            instance=instance,
        )

        assert result.success
        # Metadata should be logged since feature data exists
        event = result.get_asset_materialization_events()[0]
        metadata = event.step_materialization_data.materialization.metadata
        assert "metaxy/feature" in metadata
        assert "metaxy/info" in metadata
        assert "metaxy/store" in metadata

    def test_handle_output_with_none_no_metadata_when_data_missing(
        self,
        upstream_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that handle_output with None doesn't log metadata when feature data doesn't exist."""

        @dg.asset(
            metadata={"metaxy/feature": "features/upstream"},
            io_manager_key="metaxy_io_manager",
        )
        def my_asset():
            # Return None and no data exists - no metadata should be logged
            return None

        result = dg.materialize(
            [my_asset],
            resources=resources,
            instance=instance,
        )

        assert result.success
        # No metadata should be logged when feature data doesn't exist
        event = result.get_asset_materialization_events()[0]
        metadata = event.step_materialization_data.materialization.metadata
        assert "metaxy/feature" not in metadata
        assert "metaxy/info" not in metadata
        assert "metaxy/store" not in metadata

    def test_handle_output_with_none_logs_metadata_when_written_in_asset(
        self,
        upstream_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that handle_output with None logs metadata when data is written inside the asset."""

        @dg.asset(
            metadata={"metaxy/feature": "features/upstream"},
            io_manager_key="metaxy_io_manager",
        )
        def my_asset(store: dg.ResourceParam[mx.MetadataStore]):
            # Write data directly to store inside the asset
            with store.open("write"):
                store.write_metadata(
                    upstream_feature,
                    pl.DataFrame(
                        {
                            "id": ["in_asset_1", "in_asset_2"],
                            "metaxy_provenance_by_field": [
                                {"value": "v1"},
                                {"value": "v2"},
                            ],
                        }
                    ),
                )
            # Return None - IOManager should still log metadata from the data we just wrote
            return None

        result = dg.materialize(
            [my_asset],
            resources=resources,
            instance=instance,
        )

        assert result.success
        # Metadata should be logged since feature data was written inside the asset
        event = result.get_asset_materialization_events()[0]
        metadata = event.step_materialization_data.materialization.metadata
        assert "metaxy/feature" in metadata
        assert metadata["metaxy/feature"].value == "features/upstream"
        assert "metaxy/info" in metadata
        assert "metaxy/store" in metadata
        # Should have 2 rows
        assert metadata["dagster/row_count"].value == 2

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

    def test_load_input_from_fallback_store(
        self,
        upstream_feature: type[mx.BaseFeature],
        downstream_feature: type[mx.BaseFeature],
        tmp_path,
        instance: dg.DagsterInstance,
    ):
        """Test that load_input correctly reads from fallback store and attaches correct metadata.

        This tests the fix for issue #549: when a feature is only in a fallback store,
        the IO manager should read from the fallback and attach the fallback store's
        metadata to the LOADED_INPUT event.
        """
        import metaxy.ext.dagster as mxd
        from metaxy.metadata_store.delta import DeltaMetadataStore

        # Create fallback and primary stores
        fallback_path = tmp_path / "fallback"
        primary_path = tmp_path / "primary"

        fallback_store = DeltaMetadataStore(root_path=fallback_path)

        # Write upstream data to fallback store only
        with fallback_store.open("write"):
            fallback_store.write_metadata(
                upstream_feature,
                pl.DataFrame(
                    {
                        "id": ["f1", "f2"],
                        "metaxy_provenance_by_field": [
                            {"value": "v1"},
                            {"value": "v2"},
                        ],
                    }
                ),
            )

        # Configure MetaxyConfig with primary store having fallback
        fallback_store_config = mx.StoreConfig(
            type="metaxy.metadata_store.delta.DeltaMetadataStore",
            config={"root_path": str(fallback_path)},
        )
        primary_store_config = mx.StoreConfig(
            type="metaxy.metadata_store.delta.DeltaMetadataStore",
            config={
                "root_path": str(primary_path),
                "fallback_stores": ["fallback"],
            },
        )

        with mx.MetaxyConfig(
            stores={"dev": primary_store_config, "fallback": fallback_store_config}
        ).use():
            store_resource = mxd.MetaxyStoreFromConfigResource(name="dev")
            resources = {
                "store": store_resource,
                "metaxy_io_manager": mxd.MetaxyIOManager(store=store_resource),
            }

            # Track what data was loaded
            captured_data = {}

            @dg.asset(
                metadata={"metaxy/feature": "features/upstream"},
                io_manager_key="metaxy_io_manager",
            )
            def upstream_asset():
                # Return None - data already exists in fallback store
                return None

            @dg.asset(
                metadata={"metaxy/feature": "features/downstream"},
                io_manager_key="metaxy_io_manager",
            )
            def downstream_asset(upstream_asset: nw.LazyFrame):  # pyright: ignore[reportMissingTypeArgument]
                # This receives data loaded via IO manager from fallback store
                data = upstream_asset.collect()
                captured_data["rows"] = len(data)
                captured_data["ids"] = set(nw.to_native(data["id"]).to_list())
                return None

            # First materialize upstream (no-op, data in fallback)
            result1 = dg.materialize(
                assets=[upstream_asset],
                resources=resources,
                instance=instance,
            )
            assert result1.success

            # Materialize downstream - this will load upstream from fallback via IO manager
            result2 = dg.materialize(
                assets=[upstream_asset, downstream_asset],
                resources=resources,
                instance=instance,
                selection=[downstream_asset],
            )
            assert result2.success

            # Verify downstream received data from fallback store
            assert captured_data["rows"] == 2
            assert captured_data["ids"] == {"f1", "f2"}

            # Find the LOADED_INPUT event for the upstream_asset input
            loaded_input_events = [
                e for e in result2.all_events if e.event_type_value == "LOADED_INPUT"
            ]
            assert len(loaded_input_events) == 1

            # Check the input metadata shows the fallback store
            loaded_input_data = loaded_input_events[0].event_specific_data
            input_metadata = loaded_input_data.metadata  # pyright: ignore
            assert "metaxy/store" in input_metadata
            # The store display should reference the fallback path
            store_display = input_metadata["metaxy/store"].value
            assert "fallback" in store_display  # pyright: ignore
            assert "primary" not in store_display  # pyright: ignore

            # Also verify dagster/uri points to fallback
            assert "dagster/uri" in input_metadata
            uri_value = input_metadata["dagster/uri"].value
            assert "fallback" in str(uri_value)

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
        # First write data externally so metadata can be read
        store = mx.MetaxyConfig.get().get_store("dev")
        with store.open("write"):
            store.write_metadata(
                upstream_feature,
                pl.DataFrame(
                    {
                        "id": ["a", "b"],
                        "metaxy_provenance_by_field": [
                            {"value": "v1"},
                            {"value": "v2"},
                        ],
                    }
                ),
            )

        @dg.asset(
            metadata={"metaxy/feature": "features/upstream"},
            io_manager_key="metaxy_io_manager",
        )
        def my_asset():
            # Return None - data was written externally
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
        assert "feature" in info
        assert "metaxy" in info
        # Feature info
        feature_info = info["feature"]
        assert isinstance(feature_info, dict)
        assert "project" in feature_info
        assert "spec" in feature_info
        assert "version" in feature_info
        assert "type" in feature_info
        # Metaxy info
        metaxy_info = info["metaxy"]
        assert isinstance(metaxy_info, dict)
        assert "version" in metaxy_info
        assert "plugins" in metaxy_info

        # Check metaxy/store contains store type info
        store_meta = metadata["metaxy/store"].value
        assert isinstance(store_meta, dict)

        # Get the actual store from config to verify values
        actual_store = mx.MetaxyConfig.get().get_store("dev")
        store_cls = actual_store.__class__

        # Type should be fully qualified class name
        expected_type = f"{store_cls.__module__}.{store_cls.__qualname__}"
        assert store_meta["type"] == expected_type

        # Display should match store.display()
        assert store_meta["display"] == actual_store.display()

        # versioning_engine should match
        assert store_meta["versioning_engine"] == actual_store._versioning_engine

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

    def test_partitioned_load_input_filters_to_single_partition(
        self,
        partitions_def: dg.StaticPartitionsDefinition,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that load_input via IO manager filters to the current partition.

        This test verifies that when a partitioned downstream asset loads data
        from a partitioned upstream asset via the IO manager (using type annotation),
        the data is filtered to only the current partition.
        """
        # Create upstream feature
        upstream_spec = mx.FeatureSpec(
            key=["features", "upstream_partitioned"],
            id_columns=["id"],
            fields=["value"],
        )

        class UpstreamPartitioned(mx.BaseFeature, spec=upstream_spec):
            id: str

        # Create downstream feature
        downstream_spec = mx.FeatureSpec(
            key=["features", "downstream_partitioned"],
            id_columns=["id"],
            fields=["result"],
            deps=[mx.FeatureDep(feature=UpstreamPartitioned)],
        )

        class DownstreamPartitioned(mx.BaseFeature, spec=downstream_spec):
            id: str

        captured_data: dict[str, Any] = {}

        @dg.asset(
            metadata={
                "metaxy/feature": "features/upstream_partitioned",
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
                "metaxy/feature": "features/downstream_partitioned",
                "partition_by": "partition",
            },
            io_manager_key="metaxy_io_manager",
            partitions_def=partitions_def,
        )
        def downstream_partitioned(
            context: dg.AssetExecutionContext,
            upstream_partitioned: nw.LazyFrame,  # pyright: ignore[reportMissingTypeArgument]
        ):
            # This receives data loaded via IO manager's load_input method
            # It should be filtered to the current partition
            data = upstream_partitioned.collect()
            partition = context.partition_key
            captured_data["partition"] = partition
            captured_data["row_count"] = len(data)
            captured_data["ids"] = set(data["id"].to_list())
            captured_data["partitions"] = set(data["partition"].to_list())
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
            all_data = store.read_metadata(UpstreamPartitioned).collect()
            assert len(all_data) == 6

        # Materialize downstream for partition "b" only
        # The IO manager's load_input should filter upstream data to partition "b"
        result = dg.materialize(
            [upstream_partitioned, downstream_partitioned],
            resources=resources,
            instance=instance,
            partition_key="b",
            selection=[downstream_partitioned],
        )
        assert result.success

        # Verify downstream received ONLY partition "b" data via IO manager
        assert captured_data["partition"] == "b"
        assert captured_data["row_count"] == 2, (
            f"Expected 2 rows for partition 'b', got {captured_data['row_count']}. "
            f"IDs received: {captured_data['ids']}"
        )
        assert captured_data["ids"] == {"b_1", "b_2"}
        assert captured_data["partitions"] == {"b"}

    def test_partitioned_upstream_to_unpartitioned_downstream_via_load_input(
        self,
        partitions_def: dg.StaticPartitionsDefinition,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Unpartitioned downstream loading from partitioned upstream via IO manager gets all data.

        When an unpartitioned asset loads from a partitioned upstream via the IO manager,
        it should receive ALL data (no partition filtering applied).
        """
        # Create upstream feature
        upstream_spec = mx.FeatureSpec(
            key=["features", "upstream_part_to_unpart"],
            id_columns=["id"],
            fields=["value"],
        )

        class UpstreamPartToUnpart(mx.BaseFeature, spec=upstream_spec):
            id: str

        # Create downstream feature (no deps needed for this test)
        downstream_spec = mx.FeatureSpec(
            key=["features", "downstream_unpart"],
            id_columns=["id"],
            fields=["result"],
        )

        class DownstreamUnpart(mx.BaseFeature, spec=downstream_spec):
            id: str

        captured_data: dict[str, Any] = {}

        @dg.asset(
            metadata={
                "metaxy/feature": "features/upstream_part_to_unpart",
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
            metadata={"metaxy/feature": "features/downstream_unpart"},
            io_manager_key="metaxy_io_manager",
            ins={
                "upstream_partitioned": dg.AssetIn(
                    partition_mapping=dg.AllPartitionMapping()
                )
            },
        )
        def downstream_unpartitioned(
            upstream_partitioned: nw.LazyFrame,  # pyright: ignore[reportMissingTypeArgument]
        ):
            # This receives data loaded via IO manager's load_input method
            # Since downstream is unpartitioned, it should get ALL data
            data = upstream_partitioned.collect()
            captured_data["row_count"] = len(data)
            captured_data["ids"] = set(data["id"].to_list())
            captured_data["partitions"] = set(data["partition"].to_list())
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
            all_data = store.read_metadata(UpstreamPartToUnpart).collect()
            assert len(all_data) == 6

        # Materialize unpartitioned downstream
        # The IO manager's load_input should NOT filter (no partition context)
        result = dg.materialize(
            [upstream_partitioned, downstream_unpartitioned],
            resources=resources,
            instance=instance,
            selection=[downstream_unpartitioned],
        )
        assert result.success

        # Verify downstream received ALL data via IO manager
        assert captured_data["row_count"] == 6, (
            f"Expected 6 rows (all partitions), got {captured_data['row_count']}. "
            f"IDs received: {captured_data['ids']}"
        )
        assert captured_data["ids"] == {"a_1", "a_2", "b_1", "b_2", "c_1", "c_2"}
        assert captured_data["partitions"] == {"a", "b", "c"}

    def test_unpartitioned_upstream_to_partitioned_downstream_via_load_input(
        self,
        partitions_def: dg.StaticPartitionsDefinition,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Partitioned downstream loading from unpartitioned upstream via IO manager gets all data.

        When a partitioned asset loads from an unpartitioned upstream via the IO manager,
        it should receive ALL upstream data (upstream has no partition_by, so no filtering).
        """
        # Create upstream feature (unpartitioned)
        upstream_spec = mx.FeatureSpec(
            key=["features", "upstream_unpart"],
            id_columns=["id"],
            fields=["value"],
        )

        class UpstreamUnpart(mx.BaseFeature, spec=upstream_spec):
            id: str

        # Create downstream feature
        downstream_spec = mx.FeatureSpec(
            key=["features", "downstream_part_from_unpart"],
            id_columns=["id"],
            fields=["result"],
        )

        class DownstreamPartFromUnpart(mx.BaseFeature, spec=downstream_spec):
            id: str

        captured_data: dict[str, Any] = {}

        @dg.asset(
            metadata={"metaxy/feature": "features/upstream_unpart"},
            io_manager_key="metaxy_io_manager",
        )
        def upstream_unpartitioned():
            # Write 4 rows with no partition column
            return pl.DataFrame(
                {
                    "id": ["x_1", "x_2", "x_3", "x_4"],
                    "metaxy_provenance_by_field": [
                        {"value": "v1"},
                        {"value": "v2"},
                        {"value": "v3"},
                        {"value": "v4"},
                    ],
                }
            )

        @dg.asset(
            metadata={
                "metaxy/feature": "features/downstream_part_from_unpart",
                "partition_by": "partition",
            },
            io_manager_key="metaxy_io_manager",
            partitions_def=partitions_def,
            ins={
                "upstream_unpartitioned": dg.AssetIn(
                    partition_mapping=dg.AllPartitionMapping()
                )
            },
        )
        def downstream_partitioned(
            context: dg.AssetExecutionContext,
            upstream_unpartitioned: nw.LazyFrame,  # pyright: ignore[reportMissingTypeArgument]
        ):
            # This receives data loaded via IO manager's load_input method
            # Upstream has no partition_by, so we should get all upstream data
            data = upstream_unpartitioned.collect()
            partition = context.partition_key
            captured_data["partition"] = partition
            captured_data["row_count"] = len(data)
            captured_data["ids"] = set(data["id"].to_list())
            return None

        # Materialize upstream (unpartitioned)
        result = dg.materialize(
            [upstream_unpartitioned],
            resources=resources,
            instance=instance,
        )
        assert result.success

        # Verify upstream data was written
        with mx.MetaxyConfig.get().get_store("dev") as store:
            all_data = store.read_metadata(UpstreamUnpart).collect()
            assert len(all_data) == 4

        # Materialize downstream for partition "b"
        # Since upstream has no partition_by, IO manager should return all upstream data
        result = dg.materialize(
            [upstream_unpartitioned, downstream_partitioned],
            resources=resources,
            instance=instance,
            partition_key="b",
            selection=[downstream_partitioned],
        )
        assert result.success

        # Verify downstream received ALL upstream data (no filtering on upstream)
        assert captured_data["partition"] == "b"
        assert captured_data["row_count"] == 4, (
            f"Expected 4 rows (all upstream data), got {captured_data['row_count']}. "
            f"IDs received: {captured_data['ids']}"
        )
        assert captured_data["ids"] == {"x_1", "x_2", "x_3", "x_4"}

    def test_specific_partition_mapping_via_load_input(
        self,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test SpecificPartitionsMapping loads only specified partitions via IO manager.

        When using SpecificPartitionsMapping, the downstream asset should only receive
        data from the specified upstream partitions.
        """
        # Create upstream feature
        upstream_spec = mx.FeatureSpec(
            key=["features", "upstream_specific_mapping"],
            id_columns=["id"],
            fields=["value"],
        )

        class UpstreamSpecificMapping(mx.BaseFeature, spec=upstream_spec):
            id: str

        # Create downstream feature
        downstream_spec = mx.FeatureSpec(
            key=["features", "downstream_specific_mapping"],
            id_columns=["id"],
            fields=["result"],
        )

        class DownstreamSpecificMapping(mx.BaseFeature, spec=downstream_spec):
            id: str

        # Use different partition schemes
        upstream_partitions = dg.StaticPartitionsDefinition(
            ["2024-01", "2024-02", "2024-03"]
        )
        downstream_partitions = dg.StaticPartitionsDefinition(["q1"])

        captured_data: dict[str, Any] = {}

        @dg.asset(
            metadata={
                "metaxy/feature": "features/upstream_specific_mapping",
                "partition_by": "month",
            },
            io_manager_key="metaxy_io_manager",
            partitions_def=upstream_partitions,
        )
        def upstream_monthly(context: dg.AssetExecutionContext):
            month = context.partition_key
            return pl.DataFrame(
                {
                    "id": [f"{month}_1", f"{month}_2"],
                    "month": [month, month],
                    "metaxy_provenance_by_field": [
                        {"value": "v1"},
                        {"value": "v1"},
                    ],
                }
            )

        @dg.asset(
            metadata={
                "metaxy/feature": "features/downstream_specific_mapping",
                "partition_by": "quarter",
            },
            io_manager_key="metaxy_io_manager",
            partitions_def=downstream_partitions,
            ins={
                "upstream_monthly": dg.AssetIn(
                    partition_mapping=dg.SpecificPartitionsPartitionMapping(
                        ["2024-01", "2024-02"]  # Only Q1 months (Jan, Feb)
                    )
                )
            },
        )
        def downstream_quarterly(
            context: dg.AssetExecutionContext,
            upstream_monthly: nw.LazyFrame,  # pyright: ignore[reportMissingTypeArgument]
        ):
            # This receives data loaded via IO manager's load_input method
            # Should only get data from 2024-01 and 2024-02 due to SpecificPartitionsMapping
            data = upstream_monthly.collect()
            captured_data["row_count"] = len(data)
            captured_data["ids"] = set(data["id"].to_list())
            captured_data["months"] = set(data["month"].to_list())
            return None

        # Materialize all upstream partitions
        for month in ["2024-01", "2024-02", "2024-03"]:
            result = dg.materialize(
                [upstream_monthly],
                resources=resources,
                instance=instance,
                partition_key=month,
            )
            assert result.success

        # Verify all upstream data was written (6 rows total)
        with mx.MetaxyConfig.get().get_store("dev") as store:
            all_data = store.read_metadata(UpstreamSpecificMapping).collect()
            assert len(all_data) == 6

        # Materialize downstream for partition "q1"
        # SpecificPartitionsMapping maps q1 -> [2024-01, 2024-02]
        result = dg.materialize(
            [upstream_monthly, downstream_quarterly],
            resources=resources,
            instance=instance,
            partition_key="q1",
            selection=[downstream_quarterly],
        )
        assert result.success

        # Verify downstream received only data from 2024-01 and 2024-02
        assert captured_data["row_count"] == 4, (
            f"Expected 4 rows (Jan + Feb), got {captured_data['row_count']}. "
            f"IDs received: {captured_data['ids']}"
        )
        assert captured_data["ids"] == {
            "2024-01_1",
            "2024-01_2",
            "2024-02_1",
            "2024-02_2",
        }
        assert captured_data["months"] == {"2024-01", "2024-02"}

    def test_partitioned_output_metadata_reports_correct_counts(
        self,
        partitioned_feature: type[mx.BaseFeature],
        partitions_def: dg.StaticPartitionsDefinition,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that partitioned asset output metadata reports correct row counts.

        dagster/row_count should show total rows across all partitions at time of materialization.
        dagster/partition_row_count should show rows for the current partition.

        Since materializations happen sequentially:
        - When partition "a" is materialized: 3 rows total (3 in partition a)
        - When partition "b" is materialized: 5 rows total (2 in partition b)
        - When partition "c" is materialized: 9 rows total (4 in partition c)
        """

        @dg.asset(
            metadata={
                "metaxy/feature": "features/partitioned",
                "partition_by": "partition",
            },
            io_manager_key="metaxy_io_manager",
            partitions_def=partitions_def,
        )
        def partitioned_asset(context: dg.AssetExecutionContext):
            partition = context.partition_key
            # Each partition has different number of rows
            row_counts = {"a": 3, "b": 2, "c": 4}
            count = row_counts[partition]
            return pl.DataFrame(
                {
                    "id": [f"{partition}_{i}" for i in range(count)],
                    "partition": [partition] * count,
                    "metaxy_provenance_by_field": [
                        {"value": f"v{i}"} for i in range(count)
                    ],
                }
            )

        # Materialize all partitions sequentially
        # After each materialization, check the metadata
        expected_totals = {"a": 3, "b": 5, "c": 9}  # cumulative totals
        expected_partition_counts = {"a": 3, "b": 2, "c": 4}

        for partition in ["a", "b", "c"]:
            result = dg.materialize(
                [partitioned_asset],
                resources=resources,
                instance=instance,
                partition_key=partition,
            )
            assert result.success

            # Check the materialization metadata
            event = result.get_asset_materialization_events()[0]
            metadata = event.step_materialization_data.materialization.metadata

            assert metadata["dagster/row_count"].value == expected_totals[partition]
            assert (
                metadata["dagster/partition_row_count"].value
                == expected_partition_counts[partition]
            )


class TestMultipleAssetsPerFeature:
    """Test multiple Dagster assets contributing to the same Metaxy feature."""

    @pytest.fixture
    def shared_feature(self) -> type[mx.BaseFeature]:
        """Create a feature that will be written by multiple assets."""
        spec = mx.FeatureSpec(
            key=["features", "shared"],
            id_columns=["id"],
            fields=["value", "source"],
        )

        class SharedFeature(mx.BaseFeature, spec=spec):
            id: str

        return SharedFeature

    def test_two_assets_contribute_to_same_feature(
        self,
        shared_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that two assets can write to the same Metaxy feature (append-only)."""

        @dg.asset(
            metadata={"metaxy/feature": "features/shared"},
            io_manager_key="metaxy_io_manager",
        )
        def asset_partition_a():
            """Writes partition A data to the shared feature."""
            return pl.DataFrame(
                {
                    "id": ["a1", "a2", "a3"],
                    "metaxy_provenance_by_field": [
                        {"value": "v1", "source": "h1"},
                        {"value": "v2", "source": "h2"},
                        {"value": "v3", "source": "h3"},
                    ],
                }
            )

        @dg.asset(
            metadata={"metaxy/feature": "features/shared"},
            io_manager_key="metaxy_io_manager",
        )
        def asset_partition_b():
            """Writes partition B data to the same shared feature."""
            return pl.DataFrame(
                {
                    "id": ["b1", "b2"],
                    "metaxy_provenance_by_field": [
                        {"value": "v4", "source": "h4"},
                        {"value": "v5", "source": "h5"},
                    ],
                }
            )

        # Materialize first asset
        result1 = dg.materialize(
            [asset_partition_a],
            resources=resources,
            instance=instance,
        )
        assert result1.success

        # Verify first asset's data is in the store
        with mx.MetaxyConfig.get().get_store("dev") as store:
            data_after_a = store.read_metadata(shared_feature).collect()
            assert len(data_after_a) == 3

        # Materialize second asset
        result2 = dg.materialize(
            [asset_partition_b],
            resources=resources,
            instance=instance,
        )
        assert result2.success

        # Verify both assets' data is now in the store (append-only behavior)
        with mx.MetaxyConfig.get().get_store("dev") as store:
            all_data = store.read_metadata(shared_feature).collect()
            assert len(all_data) == 5  # 3 from A + 2 from B

            # Verify all IDs are present (convert to list via Narwhals for backend compatibility)
            ids = set(all_data["id"].to_list())
            assert ids == {"a1", "a2", "a3", "b1", "b2"}

    def test_multiple_assets_with_same_feature_key(
        self,
        shared_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test multiple assets using the same feature key as asset key."""
        import metaxy.ext.dagster as mxd

        @mxd.metaxify
        @dg.asset(
            metadata={"metaxy/feature": "features/shared"},
            io_manager_key="metaxy_io_manager",
        )
        def producer_one():
            return pl.DataFrame(
                {
                    "id": ["p1_1", "p1_2"],
                    "metaxy_provenance_by_field": [
                        {"value": "x1", "source": "s1"},
                        {"value": "x2", "source": "s2"},
                    ],
                }
            )

        @mxd.metaxify
        @dg.asset(
            metadata={"metaxy/feature": "features/shared"},
            io_manager_key="metaxy_io_manager",
        )
        def producer_two():
            return pl.DataFrame(
                {
                    "id": ["p2_1", "p2_2", "p2_3"],
                    "metaxy_provenance_by_field": [
                        {"value": "y1", "source": "t1"},
                        {"value": "y2", "source": "t2"},
                        {"value": "y3", "source": "t3"},
                    ],
                }
            )

        # Both assets use the same feature key as asset key
        assert dg.AssetKey(["features", "shared"]) in producer_one.keys
        assert dg.AssetKey(["features", "shared"]) in producer_two.keys

        # Materialize both assets
        result1 = dg.materialize(
            [producer_one],
            resources=resources,
            instance=instance,
        )
        assert result1.success

        result2 = dg.materialize(
            [producer_two],
            resources=resources,
            instance=instance,
        )
        assert result2.success

        # Verify all data is in the store
        with mx.MetaxyConfig.get().get_store("dev") as store:
            all_data = store.read_metadata(shared_feature).collect()
            assert len(all_data) == 5  # 2 from producer_one + 3 from producer_two

    def test_downstream_reads_from_multi_producer_feature(
        self,
        shared_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test downstream asset can read all data from a multi-producer feature."""
        captured_data = {}

        @dg.asset(
            metadata={"metaxy/feature": "features/shared"},
            io_manager_key="metaxy_io_manager",
        )
        def source_one():
            return pl.DataFrame(
                {
                    "id": ["s1"],
                    "metaxy_provenance_by_field": [{"value": "v1", "source": "h1"}],
                }
            )

        @dg.asset(
            metadata={"metaxy/feature": "features/shared"},
            io_manager_key="metaxy_io_manager",
        )
        def source_two():
            return pl.DataFrame(
                {
                    "id": ["s2"],
                    "metaxy_provenance_by_field": [{"value": "v2", "source": "h2"}],
                }
            )

        @dg.asset(deps=[source_one, source_two])
        def consumer(store: dg.ResourceParam[mx.MetadataStore]):
            """Consumes all data from the shared feature."""
            with store:
                data = store.read_metadata(shared_feature).collect()
                captured_data["count"] = len(data)
                captured_data["ids"] = set(data["id"].to_list())

        # Materialize sources
        dg.materialize([source_one], resources=resources, instance=instance)
        dg.materialize([source_two], resources=resources, instance=instance)

        # Materialize consumer
        result = dg.materialize(
            [source_one, source_two, consumer],
            resources=resources,
            instance=instance,
            selection=[consumer],
        )
        assert result.success

        # Verify consumer saw all data
        assert captured_data["count"] == 2
        assert captured_data["ids"] == {"s1", "s2"}
