"""Tests for build_metaxy_observation_job and build_metaxy_observation_jobs."""

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
        key=["test", "obs_job", "a"],
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
        key=["test", "obs_job", "b"],
        id_columns=["id"],
        fields=["value"],
        deps=[feature_a],
    )

    class FeatureB(mx.BaseFeature, spec=spec):
        id: str

    return FeatureB


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


class TestBuildMetaxyObservationJob:
    """Tests for build_metaxy_observation_job."""

    def test_basic_observation_job(
        self,
        feature_a: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that a basic observation job can be built and executed."""
        # Write data first
        _write_feature_data(feature_a, ["1", "2", "3"], resources, instance)

        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/obs_job/a"})
        def obs_a():
            pass

        jobs = mxd.build_metaxy_observation_job(obs_a)
        assert len(jobs) == 1
        job = jobs[0]

        # Job name is derived from feature key
        assert job.name == "observe_test__obs_job__a"

        # Execute the job
        result = job.execute_in_process(resources=resources, instance=instance)
        assert result.success

        # Check observation events - asset key is feature key due to metaxify
        obs = instance.fetch_observations(
            dg.AssetKey(["test", "obs_job", "a"]), limit=1
        )
        assert len(obs.records) == 1
        obs_meta = obs.records[0].asset_observation.metadata  # pyright: ignore[reportOptionalMemberAccess]
        assert obs_meta["dagster/row_count"].value == 3

    def test_returns_list_for_multi_asset(
        self,
        feature_a: type[mx.BaseFeature],
        feature_b: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that multi-asset with multiple metaxy features returns list of jobs."""
        _write_feature_data(feature_a, ["1", "2"], resources, instance)
        _write_feature_data(feature_b, ["3"], resources, instance)

        @mxd.metaxify()
        @dg.multi_asset(
            specs=[
                dg.AssetSpec("obs_a", metadata={"metaxy/feature": "test/obs_job/a"}),
                dg.AssetSpec("obs_b", metadata={"metaxy/feature": "test/obs_job/b"}),
            ]
        )
        def multi_obs():
            pass

        jobs = mxd.build_metaxy_observation_job(multi_obs)
        assert len(jobs) == 2
        assert jobs[0].name == "observe_test__obs_job__a"
        assert jobs[1].name == "observe_test__obs_job__b"

        # Execute both jobs
        for job in jobs:
            result = job.execute_in_process(resources=resources, instance=instance)
            assert result.success

    def test_returns_single_job_for_multi_asset_with_one_metaxy_feature(
        self,
        feature_a: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that multi-asset with one metaxy feature returns single job."""
        _write_feature_data(feature_a, ["1", "2"], resources, instance)

        @mxd.metaxify()
        @dg.multi_asset(
            specs=[
                dg.AssetSpec("obs_a", metadata={"metaxy/feature": "test/obs_job/a"}),
                dg.AssetSpec("obs_b"),  # No metaxy/feature
            ]
        )
        def multi_obs():
            pass

        jobs = mxd.build_metaxy_observation_job(multi_obs)
        assert len(jobs) == 1
        assert jobs[0].name == "observe_test__obs_job__a"

        result = jobs[0].execute_in_process(resources=resources, instance=instance)
        assert result.success

    def test_custom_store_resource_key(
        self,
        feature_a: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that a custom store resource key can be used."""
        _write_feature_data(feature_a, ["1", "2"], resources, instance)

        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/obs_job/a"})
        def obs_a():
            pass

        # Use custom resource key
        custom_resources = {"my_store": resources["store"]}

        jobs = mxd.build_metaxy_observation_job(
            obs_a,
            store_resource_key="my_store",
        )
        assert len(jobs) == 1

        result = jobs[0].execute_in_process(
            resources=custom_resources, instance=instance
        )
        assert result.success

    def test_handles_missing_feature(
        self,
        feature_a: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test graceful handling when feature doesn't exist in store."""
        # Don't write any data - feature doesn't exist in store

        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": feature_a.spec().key.to_string()})
        def missing_feature():
            pass

        jobs = mxd.build_metaxy_observation_job(missing_feature)
        assert len(jobs) == 1

        result = jobs[0].execute_in_process(resources=resources, instance=instance)
        assert result.success

        obs = instance.fetch_observations(
            dg.AssetKey(["test", "obs_job", "a"]), limit=1
        )
        assert len(obs.records) == 1
        obs_meta = obs.records[0].asset_observation.metadata  # pyright: ignore[reportOptionalMemberAccess]
        assert obs_meta["dagster/row_count"].value == 0
        assert obs_meta["error"].value == "Feature not found"

    def test_extracts_partitions_def_from_asset(
        self,
        feature_a: type[mx.BaseFeature],
    ):
        """Test that partitions_def is extracted from AssetsDefinition."""
        partitions = dg.StaticPartitionsDefinition(["2024-01-01", "2024-01-02"])

        @mxd.metaxify()
        @dg.asset(
            metadata={
                "metaxy/feature": "test/obs_job/a",
                "partition_by": "partition_date",
            },
            partitions_def=partitions,
        )
        def partitioned_asset():
            pass

        jobs = mxd.build_metaxy_observation_job(partitioned_asset)
        assert len(jobs) == 1

        # Job should have the same partitions_def
        assert jobs[0].partitions_def == partitions

    def test_raises_without_metaxy_feature(self):
        """Test that ValueError is raised if asset lacks metaxy/feature."""

        @dg.asset()
        def no_feature():
            pass

        with pytest.raises(ValueError, match="no specs with 'metaxy/feature' metadata"):
            mxd.build_metaxy_observation_job(no_feature)


class TestPartitionedObservationJob:
    """Tests for partitioned observation jobs."""

    @pytest.fixture
    def partitions_def(self) -> dg.StaticPartitionsDefinition:
        """Create a static partitions definition."""
        return dg.StaticPartitionsDefinition(["2024-01-01", "2024-01-02"])

    @pytest.fixture
    def partitioned_feature(self) -> type[mx.BaseFeature]:
        """Create a feature with a partition column."""
        spec = mx.FeatureSpec(
            key=["test", "obs_job", "partitioned"],
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

    def test_partitioned_observation_job(
        self,
        partitioned_feature: type[mx.BaseFeature],
        partitions_def: dg.StaticPartitionsDefinition,
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test observation job with partitioning."""
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

        @mxd.metaxify()
        @dg.asset(
            metadata={
                "metaxy/feature": "test/obs_job/partitioned",
                "partition_by": "partition_date",
            },
            partitions_def=partitions_def,
        )
        def partitioned_obs():
            pass

        jobs = mxd.build_metaxy_observation_job(partitioned_obs)
        assert len(jobs) == 1

        # Execute for partition 2024-01-01
        result = jobs[0].execute_in_process(
            resources=resources,
            instance=instance,
            partition_key="2024-01-01",
        )
        assert result.success

        obs = instance.fetch_observations(
            dg.AssetKey(["test", "obs_job", "partitioned"]), limit=1
        )
        assert len(obs.records) == 1
        obs_meta = obs.records[0].asset_observation.metadata  # pyright: ignore[reportOptionalMemberAccess]
        # Total count across all partitions
        assert obs_meta["dagster/row_count"].value == 5
        # Partition count
        assert obs_meta["dagster/partition_row_count"].value == 3
