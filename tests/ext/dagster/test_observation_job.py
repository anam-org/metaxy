"""Tests for build_metaxy_observation_job and build_metaxy_multi_observation_job."""

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
                "metaxy_provenance_by_field": [{"value": f"v{i}"} for i in range(len(rows))],
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
        obs = instance.fetch_observations(dg.AssetKey(["test", "obs_job", "a"]), limit=1)
        assert len(obs.records) == 1
        obs_meta = obs.records[0].asset_observation.metadata  # ty: ignore[unresolved-attribute]
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

        result = jobs[0].execute_in_process(resources=custom_resources, instance=instance)
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

        obs = instance.fetch_observations(dg.AssetKey(["test", "obs_job", "a"]), limit=1)
        assert len(obs.records) == 1
        obs_meta = obs.records[0].asset_observation.metadata  # ty: ignore[unresolved-attribute]
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
                        {"value": f"v{i}", "partition_date": context.partition_key} for i in range(len(rows))
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

        obs = instance.fetch_observations(dg.AssetKey(["test", "obs_job", "partitioned"]), limit=1)
        assert len(obs.records) == 1
        obs_meta = obs.records[0].asset_observation.metadata  # ty: ignore[unresolved-attribute]
        # Total count across all partitions
        assert obs_meta["dagster/row_count"].value == 5
        # Partition count
        assert obs_meta["dagster/partition_row_count"].value == 3


class TestBuildMetaxyMultiObservationJob:
    """Tests for build_metaxy_multi_observation_job."""

    def test_basic_multi_observation_job_with_assets(
        self,
        feature_a: type[mx.BaseFeature],
        feature_b: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that a multi observation job observes multiple features using assets arg."""
        # Write data first
        _write_feature_data(feature_a, ["1", "2", "3"], resources, instance)
        _write_feature_data(feature_b, ["4", "5"], resources, instance)

        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/obs_job/a"})
        def obs_a():
            pass

        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/obs_job/b"})
        def obs_b():
            pass

        job = mxd.build_metaxy_multi_observation_job(
            name="observe_all",
            assets=[obs_a, obs_b],
        )

        assert job.name == "observe_all"
        assert "Observe 2 Metaxy assets:" in (job.description or "")
        # Description now shows asset_key → feature_key mapping
        assert "test/obs_job/a" in (job.description or "")
        assert "test/obs_job/b" in (job.description or "")

        # Execute the job
        result = job.execute_in_process(resources=resources, instance=instance)
        assert result.success

        # Check observations for both features
        obs_a_records = instance.fetch_observations(dg.AssetKey(["test", "obs_job", "a"]), limit=1)
        assert len(obs_a_records.records) == 1
        obs_a_meta = obs_a_records.records[0].asset_observation.metadata  # ty: ignore[unresolved-attribute]
        assert obs_a_meta["dagster/row_count"].value == 3

        obs_b_records = instance.fetch_observations(dg.AssetKey(["test", "obs_job", "b"]), limit=1)
        assert len(obs_b_records.records) == 1
        obs_b_meta = obs_b_records.records[0].asset_observation.metadata  # ty: ignore[unresolved-attribute]
        assert obs_b_meta["dagster/row_count"].value == 2

    def test_basic_multi_observation_job_with_selection_and_defs(
        self,
        feature_a: type[mx.BaseFeature],
        feature_b: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that a multi observation job works with asset_selection + defs."""
        # Write data first
        _write_feature_data(feature_a, ["1", "2", "3"], resources, instance)
        _write_feature_data(feature_b, ["4", "5"], resources, instance)

        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/obs_job/a"})
        def obs_a():
            pass

        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/obs_job/b"})
        def obs_b():
            pass

        my_defs = dg.Definitions(assets=[obs_a, obs_b])
        job = mxd.build_metaxy_multi_observation_job(
            name="observe_all",
            asset_selection=dg.AssetSelection.all(),
            defs=my_defs,
        )

        assert job.name == "observe_all"
        assert "Observe 2 Metaxy assets:" in (job.description or "")

        # Execute the job
        result = job.execute_in_process(resources=resources, instance=instance)
        assert result.success

    def test_with_resolved_asset_specs(
        self,
        feature_a: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that resolved AssetSpecs from Definitions work."""
        _write_feature_data(feature_a, ["1", "2"], resources, instance)

        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/obs_job/a"})
        def obs_a():
            pass

        my_defs = dg.Definitions(assets=[obs_a])
        specs = my_defs.resolve_all_asset_specs()

        job = mxd.build_metaxy_multi_observation_job(
            name="observe_specs",
            assets=specs,
        )

        result = job.execute_in_process(resources=resources, instance=instance)
        assert result.success

    def test_raises_without_metaxy_feature(self):
        """Test that ValueError is raised if no assets have metaxy/feature."""

        @dg.asset()
        def no_feature():
            pass

        with pytest.raises(ValueError, match="No assets have specs"):
            mxd.build_metaxy_multi_observation_job(
                name="observe_none",
                assets=[no_feature],
            )

    def test_handles_missing_feature(
        self,
        feature_a: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test graceful handling when feature doesn't exist in store."""
        # Don't write any data - feature doesn't exist

        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": feature_a.spec().key.to_string()})
        def missing_feature():
            pass

        job = mxd.build_metaxy_multi_observation_job(
            name="observe_missing",
            assets=[missing_feature],
        )

        result = job.execute_in_process(resources=resources, instance=instance)
        assert result.success

        obs = instance.fetch_observations(dg.AssetKey(["test", "obs_job", "a"]), limit=1)
        assert len(obs.records) == 1
        obs_meta = obs.records[0].asset_observation.metadata  # ty: ignore[unresolved-attribute]
        assert obs_meta["dagster/row_count"].value == 0
        assert obs_meta["error"].value == "Feature not found"

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

        custom_resources = {"my_store": resources["store"]}

        job = mxd.build_metaxy_multi_observation_job(
            name="observe_custom",
            assets=[obs_a],
            store_resource_key="my_store",
        )

        result = job.execute_in_process(resources=custom_resources, instance=instance)
        assert result.success

    def test_runtime_asset_keys_config_filters_observations(
        self,
        feature_a: type[mx.BaseFeature],
        feature_b: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that asset_keys config filters which assets are observed at runtime."""
        # Write data for both features
        _write_feature_data(feature_a, ["1", "2", "3"], resources, instance)
        _write_feature_data(feature_b, ["4", "5"], resources, instance)

        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/obs_job/a"})
        def obs_a():
            pass

        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/obs_job/b"})
        def obs_b():
            pass

        job = mxd.build_metaxy_multi_observation_job(
            name="observe_filtered",
            assets=[obs_a, obs_b],
        )

        # Execute with config that filters to only feature_a
        result = job.execute_in_process(
            resources=resources,
            instance=instance,
            run_config={"ops": {"observe_filtered_fanout": {"config": {"asset_keys": ["test/obs_job/a"]}}}},
        )
        assert result.success

        # Should have observation for feature_a
        obs_a_records = instance.fetch_observations(dg.AssetKey(["test", "obs_job", "a"]), limit=1)
        assert len(obs_a_records.records) == 1
        obs_a_meta = obs_a_records.records[0].asset_observation.metadata  # ty: ignore[unresolved-attribute]
        assert obs_a_meta["dagster/row_count"].value == 3

        # Should NOT have observation for feature_b (it was filtered out)
        obs_b_records = instance.fetch_observations(dg.AssetKey(["test", "obs_job", "b"]), limit=1)
        assert len(obs_b_records.records) == 0

    def test_runtime_config_with_invalid_asset_key_raises(
        self,
        feature_a: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that invalid asset keys in config raise an error."""
        _write_feature_data(feature_a, ["1", "2"], resources, instance)

        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/obs_job/a"})
        def obs_a():
            pass

        job = mxd.build_metaxy_multi_observation_job(
            name="observe_invalid",
            assets=[obs_a],
        )

        # Execute with config that includes a non-existent asset key
        result = job.execute_in_process(
            resources=resources,
            instance=instance,
            run_config={"ops": {"observe_invalid_fanout": {"config": {"asset_keys": ["nonexistent/asset"]}}}},
            raise_on_error=False,
        )
        assert not result.success

    def test_raises_on_mismatched_partitions_def(
        self,
        feature_a: type[mx.BaseFeature],
        feature_b: type[mx.BaseFeature],
    ):
        """Test that ValueError is raised if assets have different partitions_def."""
        partitions_a = dg.StaticPartitionsDefinition(["2024-01-01"])
        partitions_b = dg.StaticPartitionsDefinition(["2024-01-02"])

        @mxd.metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/obs_job/a"},
            partitions_def=partitions_a,
        )
        def obs_a():
            pass

        @mxd.metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/obs_job/b"},
            partitions_def=partitions_b,
        )
        def obs_b():
            pass

        with pytest.raises(ValueError, match="same partitions_def"):
            mxd.build_metaxy_multi_observation_job(
                name="observe_mismatched",
                assets=[obs_a, obs_b],
            )

    def test_raises_when_both_assets_and_selection_provided(
        self,
        feature_a: type[mx.BaseFeature],
    ):
        """Test that ValueError is raised if both assets and asset_selection are provided."""

        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/obs_job/a"})
        def obs_a():
            pass

        my_defs = dg.Definitions(assets=[obs_a])

        with pytest.raises(ValueError, match="Cannot provide both"):
            mxd.build_metaxy_multi_observation_job(
                name="observe_both",
                assets=[obs_a],
                asset_selection=dg.AssetSelection.all(),
                defs=my_defs,
            )

    def test_raises_when_selection_without_defs(self):
        """Test that ValueError is raised if asset_selection is provided without defs."""
        with pytest.raises(ValueError, match="'asset_selection' requires 'defs'"):
            mxd.build_metaxy_multi_observation_job(
                name="observe_no_defs",
                asset_selection=dg.AssetSelection.all(),
            )

    def test_raises_when_defs_without_selection(self):
        """Test that ValueError is raised if defs is provided without asset_selection."""

        @dg.asset()
        def obs_a():
            pass

        my_defs = dg.Definitions(assets=[obs_a])

        with pytest.raises(ValueError, match="'defs' requires 'asset_selection'"):
            mxd.build_metaxy_multi_observation_job(
                name="observe_no_selection",
                defs=my_defs,
            )

    def test_raises_when_no_arguments_provided(self):
        """Test that ValueError is raised if neither assets nor selection is provided."""
        with pytest.raises(ValueError, match="Must provide either"):
            mxd.build_metaxy_multi_observation_job(name="observe_nothing")

    def test_multiple_jobs_in_same_definitions_have_unique_op_names(
        self,
        feature_a: type[mx.BaseFeature],
        feature_b: type[mx.BaseFeature],
        resources: dict[str, Any],
    ):
        """Test that multiple observation jobs can coexist in the same Definitions.

        Each job should have unique op names to avoid conflicts when Dagster
        validates the repository.
        """

        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/obs_job/a"})
        def obs_a():
            pass

        @mxd.metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/obs_job/b"})
        def obs_b():
            pass

        # Create two separate observation jobs
        job_1 = mxd.build_metaxy_multi_observation_job(
            name="observe_features_group_1",
            assets=[obs_a],
        )
        job_2 = mxd.build_metaxy_multi_observation_job(
            name="observe_features_group_2",
            assets=[obs_b],
        )

        # This should NOT raise DagsterInvalidDefinitionError about conflicting op names
        defs = dg.Definitions(
            assets=[obs_a, obs_b],
            jobs=[job_1, job_2],
            resources=resources,
        )

        # Getting the repository triggers validation of all jobs and their op names
        # This would raise DagsterInvalidDefinitionError if op names conflict
        repo = defs.get_repository_def()
        all_jobs = repo.get_all_jobs()

        # Verify both jobs are present
        job_names = {job.name for job in all_jobs}
        assert "observe_features_group_1" in job_names
        assert "observe_features_group_2" in job_names


class TestMultiObservationJobWithMetaxyPartition:
    """Tests for build_metaxy_multi_observation_job with metaxy/partition.

    When multiple Dagster assets contribute to the same Metaxy feature via
    metaxy/partition metadata, the observation job should spawn an op for
    each asset, not each unique feature.
    """

    @pytest.fixture
    def shared_feature(self) -> type[mx.BaseFeature]:
        """Create a feature with a region field that will be partitioned across assets."""
        spec = mx.FeatureSpec(
            key=["test", "obs_job", "shared"],
            id_columns=["id"],
            fields=["region", "value"],
        )

        class SharedFeature(mx.BaseFeature, spec=spec):
            id: str

        return SharedFeature

    def _write_partitioned_data(
        self,
        feature: type[mx.BaseFeature],
        partition_metadata: dict[str, str],
        rows: list[dict[str, Any]],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ) -> None:
        """Helper to write data with a metaxy/partition value."""
        asset_name = "_".join(partition_metadata.values())

        @mxd.metaxify()
        @dg.asset(
            name=f"write_{asset_name}",
            metadata={
                "metaxy/feature": feature.spec().key.to_string(),
                "metaxy/partition": partition_metadata,
            },
            io_manager_key="metaxy_io_manager",
        )
        def write_data():
            return pl.DataFrame(rows)

        dg.materialize([write_data], resources=resources, instance=instance)

    def test_multi_observation_job_spawns_op_per_asset_with_metaxy_partition(
        self,
        shared_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that job spawns one op per asset when assets share the same feature via metaxy/partition."""
        # Write US region data
        self._write_partitioned_data(
            shared_feature,
            {"region": "us"},
            [
                {
                    "id": "us_1",
                    "region": "us",
                    "metaxy_provenance_by_field": {"value": "v1", "region": "fixed"},
                },
                {
                    "id": "us_2",
                    "region": "us",
                    "metaxy_provenance_by_field": {"value": "v2", "region": "fixed"},
                },
            ],
            resources,
            instance,
        )

        # Write EU region data
        self._write_partitioned_data(
            shared_feature,
            {"region": "eu"},
            [
                {
                    "id": "eu_1",
                    "region": "eu",
                    "metaxy_provenance_by_field": {"value": "v3", "region": "fixed"},
                },
                {
                    "id": "eu_2",
                    "region": "eu",
                    "metaxy_provenance_by_field": {"value": "v4", "region": "fixed"},
                },
                {
                    "id": "eu_3",
                    "region": "eu",
                    "metaxy_provenance_by_field": {"value": "v5", "region": "fixed"},
                },
            ],
            resources,
            instance,
        )

        # Create two assets that both point to the same feature with different partitions
        # Note: We do NOT use @metaxify() here because metaxify() changes the asset key
        # to match the feature key, which would make both assets have the same key.
        # Instead, we keep distinct asset keys (obs_us, obs_eu) that both write to
        # the same Metaxy feature via metaxy/partition.
        @dg.asset(
            metadata={
                "metaxy/feature": "test/obs_job/shared",
                "metaxy/partition": {"region": "us"},
            },
        )
        def obs_us():
            pass

        @dg.asset(
            metadata={
                "metaxy/feature": "test/obs_job/shared",
                "metaxy/partition": {"region": "eu"},
            },
        )
        def obs_eu():
            pass

        # Build the multi observation job - should have 2 ops (one per asset)
        job = mxd.build_metaxy_multi_observation_job(
            name="observe_shared",
            assets=[obs_us, obs_eu],
        )

        # The job description should mention both assets
        assert job.description is not None
        # Even though they share the same feature key, we should see 2 assets mentioned
        # (this tests the fix - before, only 1 would be present)
        assert "Observe 2" in job.description

        # Execute the job
        result = job.execute_in_process(resources=resources, instance=instance)
        assert result.success

        # Check observations for BOTH assets
        # Each asset (obs_us, obs_eu) should have its own observation with correct row counts
        obs_us_records = instance.fetch_observations(dg.AssetKey("obs_us"), limit=10)
        assert len(obs_us_records.records) == 1, (
            f"Expected 1 observation for obs_us, but got {len(obs_us_records.records)}"
        )
        # obs_us should see only US region rows (2 rows)
        obs_us_meta = obs_us_records.records[0].asset_observation.metadata  # ty: ignore[unresolved-attribute]
        assert obs_us_meta["dagster/row_count"].value == 2, (
            f"Expected obs_us to see 2 rows (US region only), but got {obs_us_meta['dagster/row_count'].value}"
        )

        obs_eu_records = instance.fetch_observations(dg.AssetKey("obs_eu"), limit=10)
        assert len(obs_eu_records.records) == 1, (
            f"Expected 1 observation for obs_eu, but got {len(obs_eu_records.records)}"
        )
        # obs_eu should see only EU region rows (3 rows)
        obs_eu_meta = obs_eu_records.records[0].asset_observation.metadata  # ty: ignore[unresolved-attribute]
        assert obs_eu_meta["dagster/row_count"].value == 3, (
            f"Expected obs_eu to see 3 rows (EU region only), but got {obs_eu_meta['dagster/row_count'].value}"
        )

    def test_multi_observation_job_metadata_shows_both_assets(
        self,
        shared_feature: type[mx.BaseFeature],
    ):
        """Test that job metadata references both assets even when they share a feature."""

        # Note: We do NOT use @metaxify() here because it would change both asset keys
        # to be the same (the feature key)
        @dg.asset(
            metadata={
                "metaxy/feature": "test/obs_job/shared",
                "metaxy/partition": {"region": "us"},
            },
        )
        def obs_us():
            pass

        @dg.asset(
            metadata={
                "metaxy/feature": "test/obs_job/shared",
                "metaxy/partition": {"region": "eu"},
            },
        )
        def obs_eu():
            pass

        job = mxd.build_metaxy_multi_observation_job(
            name="observe_shared",
            assets=[obs_us, obs_eu],
        )

        # Job metadata should reference both assets
        assert job.metadata is not None
        asset_metadata_keys = [k for k in job.metadata if k.startswith("metaxy/asset/")]
        assert len(asset_metadata_keys) == 2, (
            f"Expected 2 asset metadata entries, got {len(asset_metadata_keys)}: {asset_metadata_keys}"
        )
