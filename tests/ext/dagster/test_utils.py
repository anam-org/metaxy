"""Tests for dagster utils: generate_materialization_events and generate_observation_events."""

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
    """Tests for generate_materialization_events."""

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
        events = list(
            mxd.generate_materialization_events(context, metadata_store, specs)
        )

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
        events = list(
            mxd.generate_materialization_events(context, metadata_store, specs)
        )

        assert len(events) == 1
        assert events[0].metadata is not None
        assert events[0].metadata["dagster/row_count"] == 3
        assert events[0].asset_key == dg.AssetKey("my_asset")
        # data_version should be set (based on mean of metaxy_created_at)
        assert events[0].data_version is not None
        assert events[0].data_version.value != "empty"

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
        events = list(
            mxd.generate_materialization_events(context, metadata_store, specs)
        )

        assert len(events) == 1
        assert events[0].asset_key == dg.AssetKey(["custom", "asset", "key"])

    def test_raises_on_missing_metadata(
        self,
        feature_a: type[mx.BaseFeature],
        metadata_store: mx.MetadataStore,
    ):
        """Test that ValueError is raised when metaxy/feature metadata is missing."""
        specs = [dg.AssetSpec("my_asset")]  # No metaxy/feature metadata

        context = dg.build_asset_context()
        with pytest.raises(ValueError, match="missing 'metaxy/feature' metadata"):
            list(mxd.generate_materialization_events(context, metadata_store, specs))


class TestGenerateObservationEvents:
    """Tests for generate_observation_events."""

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
        events = list(mxd.generate_observation_events(context, metadata_store, specs))

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
        events = list(mxd.generate_observation_events(context, metadata_store, specs))

        assert len(events) == 1
        assert events[0].metadata is not None
        assert events[0].metadata["dagster/row_count"] == 5
        # data_version should be set (based on mean of metaxy_created_at)
        assert events[0].data_version is not None
        assert events[0].data_version.value != "empty"

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
        events = list(mxd.generate_observation_events(context, metadata_store, specs))

        assert len(events) == 1
        assert events[0].asset_key == dg.AssetKey(["custom", "key"])

    def test_raises_on_missing_metadata(
        self,
        feature_a: type[mx.BaseFeature],
        metadata_store: mx.MetadataStore,
    ):
        """Test that ValueError is raised when metaxy/feature metadata is missing."""
        specs = [dg.AssetSpec("my_asset")]

        context = dg.build_asset_context()
        with pytest.raises(ValueError, match="missing 'metaxy/feature' metadata"):
            list(mxd.generate_observation_events(context, metadata_store, specs))


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
        """Test that materialization events filter by partition key."""
        # Write data for two partitions
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

        # Test with partition "2024-01-01" (3 rows)
        context = dg.build_asset_context(partition_key="2024-01-01")
        events = list(
            mxd.generate_materialization_events(context, metadata_store, specs)
        )

        assert len(events) == 1
        assert events[0].metadata is not None
        assert events[0].metadata["dagster/row_count"] == 3

        # Test with partition "2024-01-02" (2 rows)
        context = dg.build_asset_context(partition_key="2024-01-02")
        events = list(
            mxd.generate_materialization_events(context, metadata_store, specs)
        )

        assert len(events) == 1
        assert events[0].metadata is not None
        assert events[0].metadata["dagster/row_count"] == 2

    def test_observation_filters_by_partition(
        self,
        partitioned_feature: type[mx.BaseFeature],
        partitions_def: dg.StaticPartitionsDefinition,
        metadata_store: mx.MetadataStore,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that observation events filter by partition key."""
        # Write data for two partitions
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

        # Test with partition "2024-01-01" (2 rows)
        context = dg.build_asset_context(partition_key="2024-01-01")
        events = list(mxd.generate_observation_events(context, metadata_store, specs))

        assert len(events) == 1
        assert events[0].metadata is not None
        assert events[0].metadata["dagster/row_count"] == 2

        # Test with partition "2024-01-02" (4 rows)
        context = dg.build_asset_context(partition_key="2024-01-02")
        events = list(mxd.generate_observation_events(context, metadata_store, specs))

        assert len(events) == 1
        assert events[0].metadata is not None
        assert events[0].metadata["dagster/row_count"] == 4

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
        events = list(
            mxd.generate_materialization_events(context, metadata_store, specs)
        )

        assert len(events) == 1
        assert events[0].metadata is not None
        # Without partition_by, should return all 5 rows
        assert events[0].metadata["dagster/row_count"] == 5
