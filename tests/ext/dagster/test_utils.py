"""Tests for dagster utils: generate_materialize_results and generate_observe_results."""

from typing import Any

import dagster as dg
import polars as pl
import pytest

import metaxy as mx
import metaxy.ext.dagster as mxd


@pytest.fixture
def metadata_store() -> mx.MetadataStore:
    """Get the MetadataStore from the current config."""
    return mx.MetaxyConfig.get().get_store("dev")


@pytest.fixture
def feature_a() -> type[mx.BaseFeature]:
    """Create feature A (no dependencies)."""
    spec = mx.FeatureSpec(
        key=["test", "utils", "a"],
        id_columns=["id"],
        fields=["value"],
    )

    class FeatureA(mx.BaseFeature, spec=spec):
        id: str

    return FeatureA


@pytest.fixture
def feature_b(feature_a: type[mx.BaseFeature]) -> type[mx.BaseFeature]:
    """Create feature B (depends on A)."""
    spec = mx.FeatureSpec(
        key=["test", "utils", "b"],
        id_columns=["id"],
        fields=["value"],
        deps=[feature_a],
    )

    class FeatureB(mx.BaseFeature, spec=spec):
        id: str

    return FeatureB


@pytest.fixture
def feature_c(feature_b: type[mx.BaseFeature]) -> type[mx.BaseFeature]:
    """Create feature C (depends on B, so transitively on A)."""
    spec = mx.FeatureSpec(
        key=["test", "utils", "c"],
        id_columns=["id"],
        fields=["value"],
        deps=[feature_b],
    )

    class FeatureC(mx.BaseFeature, spec=spec):
        id: str

    return FeatureC


def _write_feature_data(
    feature: type[mx.BaseFeature],
    rows: list[str],
    resources: dict[str, Any],
    instance: dg.DagsterInstance,
) -> None:
    """Helper to write data to a feature."""

    @mxd.metaxify()
    @dg.asset(
        metadata={"metaxy/feature": feature.spec().key.to_string()},
        io_manager_key="metaxy_io_manager",
    )
    def write_data():
        return pl.DataFrame(
            {
                "id": rows,
                "metaxy_provenance_by_field": [
                    {"value": f"v{i}"} for i in range(len(rows))
                ],
            }
        )

    dg.materialize([write_data], resources=resources, instance=instance)


class TestGenerateMaterializationEvents:
    """Tests for generate_materialize_results."""

    def test_yields_events_in_topological_order(
        self,
        feature_a: type[mx.BaseFeature],
        feature_b: type[mx.BaseFeature],
        feature_c: type[mx.BaseFeature],
        metadata_store: mx.MetadataStore,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that events are yielded in topological order (dependencies first)."""
        # Write data for all features
        _write_feature_data(feature_a, ["a1", "a2"], resources, instance)
        _write_feature_data(feature_b, ["b1", "b2"], resources, instance)
        _write_feature_data(feature_c, ["c1", "c2"], resources, instance)

        # Create AssetSpecs in reverse order (C, B, A)
        specs = [
            dg.AssetSpec("asset_c", metadata={"metaxy/feature": "test/utils/c"}),
            dg.AssetSpec("asset_b", metadata={"metaxy/feature": "test/utils/b"}),
            dg.AssetSpec("asset_a", metadata={"metaxy/feature": "test/utils/a"}),
        ]

        context = dg.build_asset_context()
        events = list(mxd.generate_materialize_results(context, metadata_store, specs))

        assert len(events) == 3

        # Check order: A before B before C (topological order)
        asset_keys = [e.asset_key for e in events]
        assert asset_keys == [
            dg.AssetKey("asset_a"),
            dg.AssetKey("asset_b"),
            dg.AssetKey("asset_c"),
        ]

    def test_includes_row_count_and_data_version(
        self,
        feature_a: type[mx.BaseFeature],
        metadata_store: mx.MetadataStore,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that each event includes dagster/row_count metadata and data_version."""
        # Write 3 rows
        _write_feature_data(feature_a, ["1", "2", "3"], resources, instance)

        specs = [dg.AssetSpec("my_asset", metadata={"metaxy/feature": "test/utils/a"})]

        context = dg.build_asset_context()
        events = list(mxd.generate_materialize_results(context, metadata_store, specs))

        assert len(events) == 1
        assert events[0].metadata is not None
        assert events[0].metadata["dagster/row_count"] == 3
        assert events[0].asset_key == dg.AssetKey("my_asset")
        # data_version should be set (based on mean of metaxy_created_at)
        assert events[0].data_version is not None
        assert events[0].data_version.value != "empty"

    def test_includes_runtime_metadata(
        self,
        feature_a: type[mx.BaseFeature],
        metadata_store: mx.MetadataStore,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that each event includes metaxy/feature, metaxy/info, and metaxy/store runtime metadata."""
        _write_feature_data(feature_a, ["1", "2"], resources, instance)

        specs = [dg.AssetSpec("my_asset", metadata={"metaxy/feature": "test/utils/a"})]

        context = dg.build_asset_context()
        events = list(mxd.generate_materialize_results(context, metadata_store, specs))

        assert len(events) == 1
        metadata = events[0].metadata
        assert metadata is not None

        # Check metaxy/feature
        assert metadata["metaxy/feature"] == "test/utils/a"

        # Check metaxy/info structure
        info = metadata["metaxy/info"]
        assert isinstance(info, dict)
        assert "feature" in info
        assert "metaxy" in info
        assert info["feature"]["project"] == mx.MetaxyConfig.get().project
        assert info["feature"]["version"] == feature_a.feature_version()
        assert info["metaxy"]["version"] == mx.__version__
        assert info["metaxy"]["plugins"] == mx.MetaxyConfig.get().plugins

        # Check metaxy/store matches actual store
        store_meta = metadata["metaxy/store"]
        assert isinstance(store_meta, dict)
        store_cls = metadata_store.__class__
        assert store_meta["type"] == f"{store_cls.__module__}.{store_cls.__qualname__}"
        assert store_meta["display"] == metadata_store.display()
        assert store_meta["versioning_engine"] == metadata_store._versioning_engine

    def test_uses_asset_spec_key(
        self,
        feature_a: type[mx.BaseFeature],
        metadata_store: mx.MetadataStore,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that the asset key from the spec is used, not the feature key."""
        _write_feature_data(feature_a, ["1"], resources, instance)

        # Use a custom asset key different from feature key
        specs = [
            dg.AssetSpec(
                ["custom", "asset", "key"],
                metadata={"metaxy/feature": "test/utils/a"},
            )
        ]

        context = dg.build_asset_context()
        events = list(mxd.generate_materialize_results(context, metadata_store, specs))

        assert len(events) == 1
        assert events[0].asset_key == dg.AssetKey(["custom", "asset", "key"])

    def test_skips_assets_without_metaxy_feature_metadata(
        self,
        feature_a: type[mx.BaseFeature],
        metadata_store: mx.MetadataStore,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that assets without metaxy/feature metadata are skipped."""
        specs = [dg.AssetSpec("my_asset")]  # No metaxy/feature metadata

        context = dg.build_asset_context()
        results = list(mxd.generate_materialize_results(context, metadata_store, specs))
        assert results == []

    def test_skips_features_not_in_store(
        self,
        feature_a: type[mx.BaseFeature],
        metadata_store: mx.MetadataStore,
    ):
        """Test that features not in store are skipped with logged exception."""
        # Don't write any data to the store - feature_a table doesn't exist
        specs = [
            dg.AssetSpec(
                "my_asset",
                metadata={"metaxy/feature": feature_a.spec().key.to_string()},
            )
        ]

        context = dg.build_asset_context()
        # Should not raise, just skip the feature and log
        results = list(mxd.generate_materialize_results(context, metadata_store, specs))
        assert results == []


class TestGenerateObservationEvents:
    """Tests for generate_observe_results."""

    def test_yields_events_in_topological_order(
        self,
        feature_a: type[mx.BaseFeature],
        feature_b: type[mx.BaseFeature],
        feature_c: type[mx.BaseFeature],
        metadata_store: mx.MetadataStore,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that events are yielded in topological order (dependencies first)."""
        # Write data for all features
        _write_feature_data(feature_a, ["a1", "a2"], resources, instance)
        _write_feature_data(feature_b, ["b1", "b2"], resources, instance)
        _write_feature_data(feature_c, ["c1", "c2"], resources, instance)

        # Create AssetSpecs in reverse order
        specs = [
            dg.AssetSpec("asset_c", metadata={"metaxy/feature": "test/utils/c"}),
            dg.AssetSpec("asset_b", metadata={"metaxy/feature": "test/utils/b"}),
            dg.AssetSpec("asset_a", metadata={"metaxy/feature": "test/utils/a"}),
        ]

        context = dg.build_asset_context()
        events = list(mxd.generate_observe_results(context, metadata_store, specs))

        assert len(events) == 3

        # Check order: A before B before C
        asset_keys = [e.asset_key for e in events]
        assert asset_keys == [
            dg.AssetKey("asset_a"),
            dg.AssetKey("asset_b"),
            dg.AssetKey("asset_c"),
        ]

    def test_includes_row_count_and_data_version(
        self,
        feature_a: type[mx.BaseFeature],
        metadata_store: mx.MetadataStore,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that each event includes dagster/row_count metadata and data_version."""
        # Write 5 rows
        _write_feature_data(feature_a, ["1", "2", "3", "4", "5"], resources, instance)

        specs = [dg.AssetSpec("my_asset", metadata={"metaxy/feature": "test/utils/a"})]

        context = dg.build_asset_context()
        events = list(mxd.generate_observe_results(context, metadata_store, specs))

        assert len(events) == 1
        assert events[0].metadata is not None
        assert events[0].metadata["dagster/row_count"] == 5
        # data_version should be set (based on mean of metaxy_created_at)
        assert events[0].data_version is not None
        assert events[0].data_version.value != "empty"

    def test_includes_runtime_metadata(
        self,
        feature_a: type[mx.BaseFeature],
        metadata_store: mx.MetadataStore,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that each event includes metaxy/feature, metaxy/info, and metaxy/store runtime metadata."""
        _write_feature_data(feature_a, ["1", "2"], resources, instance)

        specs = [dg.AssetSpec("my_asset", metadata={"metaxy/feature": "test/utils/a"})]

        context = dg.build_asset_context()
        events = list(mxd.generate_observe_results(context, metadata_store, specs))

        assert len(events) == 1
        metadata = events[0].metadata
        assert metadata is not None

        # Check metaxy/feature
        assert metadata["metaxy/feature"] == "test/utils/a"

        # Check metaxy/info structure
        info = metadata["metaxy/info"]
        assert isinstance(info, dict)
        assert "feature" in info
        assert "metaxy" in info
        assert info["feature"]["project"] == mx.MetaxyConfig.get().project
        assert info["feature"]["version"] == feature_a.feature_version()
        assert info["metaxy"]["version"] == mx.__version__
        assert info["metaxy"]["plugins"] == mx.MetaxyConfig.get().plugins

        # Check metaxy/store matches actual store
        store_meta = metadata["metaxy/store"]
        assert isinstance(store_meta, dict)
        store_cls = metadata_store.__class__
        assert store_meta["type"] == f"{store_cls.__module__}.{store_cls.__qualname__}"
        assert store_meta["display"] == metadata_store.display()
        assert store_meta["versioning_engine"] == metadata_store._versioning_engine

    def test_uses_asset_spec_key(
        self,
        feature_a: type[mx.BaseFeature],
        metadata_store: mx.MetadataStore,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that the asset key from the spec is used."""
        _write_feature_data(feature_a, ["1"], resources, instance)

        specs = [
            dg.AssetSpec(
                ["custom", "key"],
                metadata={"metaxy/feature": "test/utils/a"},
            )
        ]

        context = dg.build_asset_context()
        events = list(mxd.generate_observe_results(context, metadata_store, specs))

        assert len(events) == 1
        assert events[0].asset_key == dg.AssetKey(["custom", "key"])

    def test_skips_assets_without_metaxy_feature_metadata(
        self,
        feature_a: type[mx.BaseFeature],
        metadata_store: mx.MetadataStore,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that assets without metaxy/feature metadata are skipped."""
        specs = [dg.AssetSpec("my_asset")]

        context = dg.build_asset_context()
        results = list(mxd.generate_observe_results(context, metadata_store, specs))
        assert results == []

    def test_skips_features_not_in_store(
        self,
        feature_a: type[mx.BaseFeature],
        metadata_store: mx.MetadataStore,
    ):
        """Test that features not in store are skipped with logged exception."""
        # Don't write any data to the store - feature_a table doesn't exist
        specs = [
            dg.AssetSpec(
                "my_asset",
                metadata={"metaxy/feature": feature_a.spec().key.to_string()},
            )
        ]

        context = dg.build_asset_context()
        # Should not raise, just skip the feature and log
        results = list(mxd.generate_observe_results(context, metadata_store, specs))
        assert results == []


class TestPartitionedAssets:
    """Tests for partitioned assets."""

    @pytest.fixture
    def partitions_def(self) -> dg.StaticPartitionsDefinition:
        """Create a static partitions definition."""
        return dg.StaticPartitionsDefinition(["2024-01-01", "2024-01-02"])

    @pytest.fixture
    def partitioned_feature(self) -> type[mx.BaseFeature]:
        """Create a feature with a partition column."""
        spec = mx.FeatureSpec(
            key=["test", "utils", "partitioned"],
            id_columns=["id"],
            fields=["value", "partition_date"],
        )

        class PartitionedFeature(mx.BaseFeature, spec=spec):
            id: str

        return PartitionedFeature

    def _write_partitioned_data(
        self,
        feature: type[mx.BaseFeature],
        partition_date: str,
        rows: list[str],
        partitions_def: dg.StaticPartitionsDefinition,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ) -> None:
        """Helper to write data with a partition value."""

        @mxd.metaxify()
        @dg.asset(
            metadata={
                "metaxy/feature": feature.spec().key.to_string(),
                "partition_by": "partition_date",
            },
            io_manager_key="metaxy_io_manager",
            partitions_def=partitions_def,
        )
        def write_data(context: dg.AssetExecutionContext):
            return pl.DataFrame(
                {
                    "id": rows,
                    "partition_date": [context.partition_key] * len(rows),
                    "metaxy_provenance_by_field": [
                        {"value": f"v{i}", "partition_date": context.partition_key}
                        for i in range(len(rows))
                    ],
                }
            )

        dg.materialize(
            [write_data],
            resources=resources,
            instance=instance,
            partition_key=partition_date,
        )

    def test_materialization_filters_by_partition(
        self,
        partitioned_feature: type[mx.BaseFeature],
        partitions_def: dg.StaticPartitionsDefinition,
        metadata_store: mx.MetadataStore,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that materialization events filter by partition key.

        dagster/row_count should show total rows across all partitions.
        dagster/partition_row_count should show rows for the current partition.
        """
        # Write data for two partitions (5 total rows)
        self._write_partitioned_data(
            partitioned_feature,
            "2024-01-01",
            ["a", "b", "c"],
            partitions_def,
            resources,
            instance,
        )
        self._write_partitioned_data(
            partitioned_feature,
            "2024-01-02",
            ["d", "e"],
            partitions_def,
            resources,
            instance,
        )

        specs = [
            dg.AssetSpec(
                "partitioned_asset",
                metadata={
                    "metaxy/feature": "test/utils/partitioned",
                    "partition_by": "partition_date",
                },
            )
        ]

        # Test with partition "2024-01-01" (3 rows in partition, 5 total)
        context = dg.build_asset_context(partition_key="2024-01-01")
        events = list(mxd.generate_materialize_results(context, metadata_store, specs))

        assert len(events) == 1
        assert events[0].metadata is not None
        assert (
            events[0].metadata["dagster/row_count"] == 5
        )  # total across all partitions
        assert events[0].metadata["dagster/partition_row_count"] == 3

        # Test with partition "2024-01-02" (2 rows in partition, 5 total)
        context = dg.build_asset_context(partition_key="2024-01-02")
        events = list(mxd.generate_materialize_results(context, metadata_store, specs))

        assert len(events) == 1
        assert events[0].metadata is not None
        assert (
            events[0].metadata["dagster/row_count"] == 5
        )  # total across all partitions
        assert events[0].metadata["dagster/partition_row_count"] == 2

    def test_observation_filters_by_partition(
        self,
        partitioned_feature: type[mx.BaseFeature],
        partitions_def: dg.StaticPartitionsDefinition,
        metadata_store: mx.MetadataStore,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that observation events filter by partition key.

        dagster/row_count should show total rows across all partitions.
        dagster/partition_row_count should show rows for the current partition.
        """
        # Write data for two partitions (6 total rows)
        self._write_partitioned_data(
            partitioned_feature,
            "2024-01-01",
            ["a", "b"],
            partitions_def,
            resources,
            instance,
        )
        self._write_partitioned_data(
            partitioned_feature,
            "2024-01-02",
            ["c", "d", "e", "f"],
            partitions_def,
            resources,
            instance,
        )

        specs = [
            dg.AssetSpec(
                "partitioned_asset",
                metadata={
                    "metaxy/feature": "test/utils/partitioned",
                    "partition_by": "partition_date",
                },
            )
        ]

        # Test with partition "2024-01-01" (2 rows in partition, 6 total)
        context = dg.build_asset_context(partition_key="2024-01-01")
        events = list(mxd.generate_observe_results(context, metadata_store, specs))

        assert len(events) == 1
        assert events[0].metadata is not None
        assert (
            events[0].metadata["dagster/row_count"] == 6
        )  # total across all partitions
        assert events[0].metadata["dagster/partition_row_count"] == 2

        # Test with partition "2024-01-02" (4 rows in partition, 6 total)
        context = dg.build_asset_context(partition_key="2024-01-02")
        events = list(mxd.generate_observe_results(context, metadata_store, specs))

        assert len(events) == 1
        assert events[0].metadata is not None
        assert (
            events[0].metadata["dagster/row_count"] == 6
        )  # total across all partitions
        assert events[0].metadata["dagster/partition_row_count"] == 4

    @pytest.mark.flaky(reruns=5)
    def test_no_partition_by_metadata_returns_all_rows(
        self,
        partitioned_feature: type[mx.BaseFeature],
        partitions_def: dg.StaticPartitionsDefinition,
        metadata_store: mx.MetadataStore,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that without partition_by metadata, all rows are counted."""
        # Write data for two partitions (total 5 rows)
        self._write_partitioned_data(
            partitioned_feature,
            "2024-01-01",
            ["a", "b", "c"],
            partitions_def,
            resources,
            instance,
        )
        self._write_partitioned_data(
            partitioned_feature,
            "2024-01-02",
            ["d", "e"],
            partitions_def,
            resources,
            instance,
        )

        # No partition_by metadata - should return all rows
        specs = [
            dg.AssetSpec(
                "partitioned_asset",
                metadata={"metaxy/feature": "test/utils/partitioned"},
            )
        ]

        context = dg.build_asset_context(partition_key="2024-01-01")
        events = list(mxd.generate_materialize_results(context, metadata_store, specs))

        assert len(events) == 1
        assert events[0].metadata is not None
        # Without partition_by, should return all 5 rows
        assert events[0].metadata["dagster/row_count"] == 5


class TestMultiAssetIntegration:
    """Integration tests using actual @multi_asset decorator."""

    def test_multi_asset_with_generate_materialize_results(
        self,
        feature_a: type[mx.BaseFeature],
        feature_b: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test generate_materialize_results within a real @multi_asset."""
        # First write some data
        _write_feature_data(feature_a, ["1", "2", "3"], resources, instance)
        _write_feature_data(feature_b, ["4", "5"], resources, instance)

        specs = [
            dg.AssetSpec("multi_a", metadata={"metaxy/feature": "test/utils/a"}),
            dg.AssetSpec("multi_b", metadata={"metaxy/feature": "test/utils/b"}),
        ]

        @dg.multi_asset(specs=specs)
        def my_multi_asset(
            context: dg.AssetExecutionContext,
            store: mxd.MetaxyStoreFromConfigResource,
        ):
            yield from mxd.generate_materialize_results(context, store, specs)

        result = dg.materialize(
            [my_multi_asset], resources=resources, instance=instance
        )
        assert result.success

        # Check materialization events via instance.fetch_materializations
        mat_a = instance.fetch_materializations(dg.AssetKey("multi_a"), limit=1)
        mat_b = instance.fetch_materializations(dg.AssetKey("multi_b"), limit=1)

        assert len(mat_a.records) == 1
        assert len(mat_b.records) == 1

        # Verify row counts
        assert (
            mat_a.records[0].asset_materialization.metadata["dagster/row_count"].value  # ty: ignore[possibly-missing-attribute]
            == 3
        )
        assert (
            mat_b.records[0].asset_materialization.metadata["dagster/row_count"].value  # ty: ignore[possibly-missing-attribute]
            == 2
        )

    def test_multi_asset_reports_materialized_in_run(
        self,
        feature_a: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that metaxy/materialized_in_run is reported when store has materialization_id."""
        # Write data in run 1 (3 rows)
        _write_feature_data(feature_a, ["r1", "r2", "r3"], resources, instance)

        # Get run_id from the first materialization (run 1)
        mat_run1 = instance.fetch_materializations(
            dg.AssetKey(["test", "utils", "a"]), limit=1
        )
        run1_id = mat_run1.records[0].run_id

        specs = [
            dg.AssetSpec("mat_run_a", metadata={"metaxy/feature": "test/utils/a"}),
        ]

        @dg.multi_asset(specs=specs)
        def my_multi_asset(
            context: dg.AssetExecutionContext,
            store: mxd.MetaxyStoreFromConfigResource,
        ):
            yield from mxd.generate_materialize_results(context, store, specs)

        # Run 2: call generate_materialize_results (no new data written in this run)
        result = dg.materialize(
            [my_multi_asset], resources=resources, instance=instance
        )
        assert result.success

        mat = instance.fetch_materializations(dg.AssetKey("mat_run_a"), limit=1)
        assert len(mat.records) == 1

        # Verify this is a different run than run 1
        run2_id = mat.records[0].run_id
        assert run2_id != run1_id

        metadata = mat.records[0].asset_materialization.metadata  # ty: ignore[possibly-missing-attribute]
        # Total row count is 3
        assert metadata["dagster/row_count"].value == 3
        # metaxy/materialized_in_run should be present since the store has materialization_id
        assert "metaxy/materialized_in_run" in metadata
        # Data was written in run 1, so materialized_in_run should be 0 for run 2
        assert metadata["metaxy/materialized_in_run"].value == 0


class TestMultiObservableSourceAssetIntegration:
    """Integration tests using actual @multi_observable_source_asset decorator."""

    def test_multi_observable_source_asset_with_generate_observe_results(
        self,
        feature_a: type[mx.BaseFeature],
        feature_b: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test generate_observe_results within a real @multi_observable_source_asset."""
        # First write some data
        _write_feature_data(feature_a, ["1", "2", "3", "4"], resources, instance)
        _write_feature_data(feature_b, ["5", "6"], resources, instance)

        specs = [
            dg.AssetSpec("obs_a", metadata={"metaxy/feature": "test/utils/a"}),
            dg.AssetSpec("obs_b", metadata={"metaxy/feature": "test/utils/b"}),
        ]

        @dg.multi_observable_source_asset(specs=specs)
        def my_observable_assets(
            context: dg.AssetExecutionContext,
            store: mxd.MetaxyStoreFromConfigResource,
        ):
            yield from mxd.generate_observe_results(context, store, specs)

        # Create definitions and run observation
        defs = dg.Definitions(
            assets=[my_observable_assets],
            resources=resources,
        )

        # Get the implicit job and execute it
        job = defs.get_implicit_global_asset_job_def()
        result = job.execute_in_process(instance=instance)
        assert result.success

        # Check observation events via instance.fetch_observations
        obs_a = instance.fetch_observations(dg.AssetKey("obs_a"), limit=1)
        obs_b = instance.fetch_observations(dg.AssetKey("obs_b"), limit=1)

        assert len(obs_a.records) == 1
        assert len(obs_b.records) == 1

        # Verify row counts
        assert (
            obs_a.records[0].asset_observation.metadata["dagster/row_count"].value == 4  # ty: ignore[possibly-missing-attribute]
        )
        assert (
            obs_b.records[0].asset_observation.metadata["dagster/row_count"].value == 2  # ty: ignore[possibly-missing-attribute]
        )


class TestPartitionedMultiAssetIntegration:
    """Integration tests for partitioned multi-assets."""

    @pytest.fixture
    def partitions_def(self) -> dg.StaticPartitionsDefinition:
        """Create a static partitions definition."""
        return dg.StaticPartitionsDefinition(["2024-01-01", "2024-01-02"])

    @pytest.fixture
    def partitioned_feature_x(self) -> type[mx.BaseFeature]:
        """Create partitioned feature X."""
        spec = mx.FeatureSpec(
            key=["test", "utils", "partitioned_x"],
            id_columns=["id"],
            fields=["value", "partition_date"],
        )

        class PartitionedFeatureX(mx.BaseFeature, spec=spec):
            id: str

        return PartitionedFeatureX

    @pytest.fixture
    def partitioned_feature_y(
        self, partitioned_feature_x: type[mx.BaseFeature]
    ) -> type[mx.BaseFeature]:
        """Create partitioned feature Y (depends on X)."""
        spec = mx.FeatureSpec(
            key=["test", "utils", "partitioned_y"],
            id_columns=["id"],
            fields=["value", "partition_date"],
            deps=[partitioned_feature_x],
        )

        class PartitionedFeatureY(mx.BaseFeature, spec=spec):
            id: str

        return PartitionedFeatureY

    def _write_partitioned_data(
        self,
        feature: type[mx.BaseFeature],
        partition_date: str,
        rows: list[str],
        partitions_def: dg.StaticPartitionsDefinition,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ) -> None:
        """Helper to write data with a partition value."""

        @mxd.metaxify()
        @dg.asset(
            metadata={
                "metaxy/feature": feature.spec().key.to_string(),
                "partition_by": "partition_date",
            },
            io_manager_key="metaxy_io_manager",
            partitions_def=partitions_def,
        )
        def write_data(context: dg.AssetExecutionContext):
            return pl.DataFrame(
                {
                    "id": rows,
                    "partition_date": [context.partition_key] * len(rows),
                    "metaxy_provenance_by_field": [
                        {"value": f"v{i}", "partition_date": context.partition_key}
                        for i in range(len(rows))
                    ],
                }
            )

        dg.materialize(
            [write_data],
            resources=resources,
            instance=instance,
            partition_key=partition_date,
        )

    def test_partitioned_multi_asset_filters_correctly(
        self,
        partitioned_feature_x: type[mx.BaseFeature],
        partitioned_feature_y: type[mx.BaseFeature],
        partitions_def: dg.StaticPartitionsDefinition,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test partitioned @multi_asset with generate_materialize_results.

        dagster/row_count should show total rows across all partitions.
        dagster/partition_row_count should show rows for the current partition.
        """
        # Write data for partition 2024-01-01: X has 3 rows, Y has 2 rows
        self._write_partitioned_data(
            partitioned_feature_x,
            "2024-01-01",
            ["x1", "x2", "x3"],
            partitions_def,
            resources,
            instance,
        )
        self._write_partitioned_data(
            partitioned_feature_y,
            "2024-01-01",
            ["y1", "y2"],
            partitions_def,
            resources,
            instance,
        )
        # Write data for partition 2024-01-02: X has 1 row, Y has 4 rows
        # Total: X=4 rows, Y=6 rows
        self._write_partitioned_data(
            partitioned_feature_x,
            "2024-01-02",
            ["x4"],
            partitions_def,
            resources,
            instance,
        )
        self._write_partitioned_data(
            partitioned_feature_y,
            "2024-01-02",
            ["y3", "y4", "y5", "y6"],
            partitions_def,
            resources,
            instance,
        )

        specs = [
            dg.AssetSpec(
                "part_multi_x",
                metadata={
                    "metaxy/feature": "test/utils/partitioned_x",
                    "partition_by": "partition_date",
                },
            ),
            dg.AssetSpec(
                "part_multi_y",
                metadata={
                    "metaxy/feature": "test/utils/partitioned_y",
                    "partition_by": "partition_date",
                },
            ),
        ]

        @dg.multi_asset(specs=specs, partitions_def=partitions_def)
        def partitioned_multi_asset(
            context: dg.AssetExecutionContext,
            store: mxd.MetaxyStoreFromConfigResource,
        ):
            yield from mxd.generate_materialize_results(context, store, specs)

        # Materialize partition 2024-01-01
        result = dg.materialize(
            [partitioned_multi_asset],
            resources=resources,
            instance=instance,
            partition_key="2024-01-01",
        )
        assert result.success

        # Fetch materializations for partition 2024-01-01
        mat_x = instance.fetch_materializations(dg.AssetKey("part_multi_x"), limit=1)
        mat_y = instance.fetch_materializations(dg.AssetKey("part_multi_y"), limit=1)

        assert len(mat_x.records) == 1
        assert len(mat_y.records) == 1
        # Total: X=4 rows, Y=6 rows
        assert (
            mat_x.records[0].asset_materialization.metadata["dagster/row_count"].value  # ty: ignore[possibly-missing-attribute]
            == 4
        )
        assert (
            mat_y.records[0].asset_materialization.metadata["dagster/row_count"].value  # ty: ignore[possibly-missing-attribute]
            == 6
        )
        # Partition 2024-01-01: X=3 rows, Y=2 rows
        assert (
            mat_x.records[0]  # ty: ignore[possibly-missing-attribute]
            .asset_materialization.metadata["dagster/partition_row_count"]
            .value
            == 3
        )
        assert (
            mat_y.records[0]  # ty: ignore[possibly-missing-attribute]
            .asset_materialization.metadata["dagster/partition_row_count"]
            .value
            == 2
        )

        # Materialize partition 2024-01-02
        result = dg.materialize(
            [partitioned_multi_asset],
            resources=resources,
            instance=instance,
            partition_key="2024-01-02",
        )
        assert result.success

        # Fetch latest materializations for partition 2024-01-02
        mat_x = instance.fetch_materializations(dg.AssetKey("part_multi_x"), limit=1)
        mat_y = instance.fetch_materializations(dg.AssetKey("part_multi_y"), limit=1)

        # Total: X=4 rows, Y=6 rows (same as before)
        assert (
            mat_x.records[0].asset_materialization.metadata["dagster/row_count"].value  # ty: ignore[possibly-missing-attribute]
            == 4
        )
        assert (
            mat_y.records[0].asset_materialization.metadata["dagster/row_count"].value  # ty: ignore[possibly-missing-attribute]
            == 6
        )
        # Partition 2024-01-02: X=1 row, Y=4 rows
        assert (
            mat_x.records[0]  # ty: ignore[possibly-missing-attribute]
            .asset_materialization.metadata["dagster/partition_row_count"]
            .value
            == 1
        )
        assert (
            mat_y.records[0]  # ty: ignore[possibly-missing-attribute]
            .asset_materialization.metadata["dagster/partition_row_count"]
            .value
            == 4
        )

    def test_partitioned_multi_observable_source_asset_filters_correctly(
        self,
        partitioned_feature_x: type[mx.BaseFeature],
        partitioned_feature_y: type[mx.BaseFeature],
        partitions_def: dg.StaticPartitionsDefinition,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test partitioned @multi_observable_source_asset with generate_observe_results.

        dagster/row_count should show total rows across all partitions.
        dagster/partition_row_count should show rows for the current partition.
        """
        # Write data for partition 2024-01-01: X has 5 rows, Y has 1 row
        self._write_partitioned_data(
            partitioned_feature_x,
            "2024-01-01",
            ["x1", "x2", "x3", "x4", "x5"],
            partitions_def,
            resources,
            instance,
        )
        self._write_partitioned_data(
            partitioned_feature_y,
            "2024-01-01",
            ["y1"],
            partitions_def,
            resources,
            instance,
        )
        # Write data for partition 2024-01-02: X has 2 rows, Y has 3 rows
        # Total: X=7 rows, Y=4 rows
        self._write_partitioned_data(
            partitioned_feature_x,
            "2024-01-02",
            ["x6", "x7"],
            partitions_def,
            resources,
            instance,
        )
        self._write_partitioned_data(
            partitioned_feature_y,
            "2024-01-02",
            ["y2", "y3", "y4"],
            partitions_def,
            resources,
            instance,
        )

        specs = [
            dg.AssetSpec(
                "part_obs_x",
                metadata={
                    "metaxy/feature": "test/utils/partitioned_x",
                    "partition_by": "partition_date",
                },
            ),
            dg.AssetSpec(
                "part_obs_y",
                metadata={
                    "metaxy/feature": "test/utils/partitioned_y",
                    "partition_by": "partition_date",
                },
            ),
        ]

        @dg.multi_observable_source_asset(specs=specs, partitions_def=partitions_def)
        def partitioned_observable_assets(
            context: dg.AssetExecutionContext,
            store: mxd.MetaxyStoreFromConfigResource,
        ):
            yield from mxd.generate_observe_results(context, store, specs)

        defs = dg.Definitions(
            assets=[partitioned_observable_assets],
            resources=resources,
        )
        job = defs.get_implicit_global_asset_job_def()

        # Observe partition 2024-01-01
        result = job.execute_in_process(
            instance=instance,
            partition_key="2024-01-01",
        )
        assert result.success

        obs_x = instance.fetch_observations(dg.AssetKey("part_obs_x"), limit=1)
        obs_y = instance.fetch_observations(dg.AssetKey("part_obs_y"), limit=1)

        assert len(obs_x.records) == 1
        assert len(obs_y.records) == 1
        # Total: X=7 rows, Y=4 rows
        assert (
            obs_x.records[0].asset_observation.metadata["dagster/row_count"].value == 7  # ty: ignore[possibly-missing-attribute]
        )
        assert (
            obs_y.records[0].asset_observation.metadata["dagster/row_count"].value == 4  # ty: ignore[possibly-missing-attribute]
        )
        # Partition 2024-01-01: X=5 rows, Y=1 row
        assert (
            obs_x.records[0]  # ty: ignore[possibly-missing-attribute]
            .asset_observation.metadata["dagster/partition_row_count"]
            .value
            == 5
        )
        assert (
            obs_y.records[0]  # ty: ignore[possibly-missing-attribute]
            .asset_observation.metadata["dagster/partition_row_count"]
            .value
            == 1
        )

        # Observe partition 2024-01-02
        result = job.execute_in_process(
            instance=instance,
            partition_key="2024-01-02",
        )
        assert result.success

        obs_x = instance.fetch_observations(dg.AssetKey("part_obs_x"), limit=1)
        obs_y = instance.fetch_observations(dg.AssetKey("part_obs_y"), limit=1)

        # Total: X=7 rows, Y=4 rows (same as before)
        assert (
            obs_x.records[0].asset_observation.metadata["dagster/row_count"].value == 7  # ty: ignore[possibly-missing-attribute]
        )
        assert (
            obs_y.records[0].asset_observation.metadata["dagster/row_count"].value == 4  # ty: ignore[possibly-missing-attribute]
        )
        # Partition 2024-01-02: X=2 rows, Y=3 rows
        assert (
            obs_x.records[0]  # ty: ignore[possibly-missing-attribute]
            .asset_observation.metadata["dagster/partition_row_count"]
            .value
            == 2
        )
        assert (
            obs_y.records[0]  # ty: ignore[possibly-missing-attribute]
            .asset_observation.metadata["dagster/partition_row_count"]
            .value
            == 3
        )
