"""Tests for metaxy/partition feature - multiple Dagster assets writing to the same Metaxy feature.

This module tests the metaxy/partition metadata feature, which allows multiple
Dagster assets to write to different logical partitions of the same Metaxy feature.
Unlike Dagster's time/date partitioning (partition_by), metaxy/partition enables
multi-producer patterns where each asset contributes a subset of data.

Test Scenario:
    - Feature with id, region (partition column), value fields
    - Asset A: metaxy/partition: {"region": "us"} → materializes 2 records
    - Asset B: metaxy/partition: {"region": "eu"} → materializes 3 records
    - Asset C: depends on A and B, observes, no partition metadata → sees 5 records
"""

from typing import Any

import dagster as dg
import polars as pl
import pytest

import metaxy as mx
import metaxy.ext.dagster as mxd


@pytest.fixture
def shared_feature() -> type[mx.BaseFeature]:
    """Create a feature with a region field that will be partitioned across assets."""
    spec = mx.FeatureSpec(
        key=["features", "shared"],
        id_columns=["id"],
        fields=["region", "value"],
    )

    class SharedFeature(mx.BaseFeature, spec=spec):
        id: str

    return SharedFeature


class TestMultiAssetPartitionMaterialization:
    """Test IOManager materialization events with metaxy/partition."""

    def test_asset_a_materializes_us_region(
        self,
        shared_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test Asset A materializes US region data with correct metadata."""

        @dg.asset(
            metadata={
                "metaxy/feature": "features/shared",
                "metaxy/partition": {"region": "us"},
            },
            io_manager_key="metaxy_io_manager",
        )
        def asset_a():
            return pl.DataFrame(
                {
                    "id": ["us_1", "us_2"],
                    "region": ["us", "us"],
                    "metaxy_provenance_by_field": [
                        {"value": "v1", "region": "fixed"},
                        {"value": "v2", "region": "fixed"},
                    ],
                }
            )

        result = dg.materialize(
            [asset_a],
            resources=resources,
            instance=instance,
        )

        assert result.success

        # Check materialization event metadata
        event = result.get_asset_materialization_events()[0]
        metadata = event.step_materialization_data.materialization.metadata

        # dagster/row_count should be 2 (only US records from this asset's view)
        assert metadata["dagster/row_count"].value == 2

        # metaxy/materialized_in_run should be 2 (Asset A wrote 2 records)
        assert metadata["metaxy/materialized_in_run"].value == 2

    def test_asset_b_materializes_eu_region(
        self,
        shared_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test Asset B materializes EU region data with correct metadata."""

        @dg.asset(
            metadata={
                "metaxy/feature": "features/shared",
                "metaxy/partition": {"region": "eu"},
            },
            io_manager_key="metaxy_io_manager",
        )
        def asset_b():
            return pl.DataFrame(
                {
                    "id": ["eu_1", "eu_2", "eu_3"],
                    "region": ["eu", "eu", "eu"],
                    "metaxy_provenance_by_field": [
                        {"value": "v3", "region": "fixed"},
                        {"value": "v4", "region": "fixed"},
                        {"value": "v5", "region": "fixed"},
                    ],
                }
            )

        result = dg.materialize(
            [asset_b],
            resources=resources,
            instance=instance,
        )

        assert result.success

        # Check materialization event metadata
        event = result.get_asset_materialization_events()[0]
        metadata = event.step_materialization_data.materialization.metadata

        # dagster/row_count should be 3 (only EU records from this asset's view)
        assert metadata["dagster/row_count"].value == 3

        # metaxy/materialized_in_run should be 3 (Asset B wrote 3 records)
        assert metadata["metaxy/materialized_in_run"].value == 3

    def test_both_assets_materialize_to_same_feature(
        self,
        shared_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that both assets write to the same feature with correct partition filtering."""

        @dg.asset(
            metadata={
                "metaxy/feature": "features/shared",
                "metaxy/partition": {"region": "us"},
            },
            io_manager_key="metaxy_io_manager",
        )
        def asset_a():
            return pl.DataFrame(
                {
                    "id": ["us_1", "us_2"],
                    "region": ["us", "us"],
                    "metaxy_provenance_by_field": [
                        {"value": "v1", "region": "fixed"},
                        {"value": "v2", "region": "fixed"},
                    ],
                }
            )

        @dg.asset(
            metadata={
                "metaxy/feature": "features/shared",
                "metaxy/partition": {"region": "eu"},
            },
            io_manager_key="metaxy_io_manager",
        )
        def asset_b():
            return pl.DataFrame(
                {
                    "id": ["eu_1", "eu_2", "eu_3"],
                    "region": ["eu", "eu", "eu"],
                    "metaxy_provenance_by_field": [
                        {"value": "v3", "region": "fixed"},
                        {"value": "v4", "region": "fixed"},
                        {"value": "v5", "region": "fixed"},
                    ],
                }
            )

        # Materialize asset_a (US region)
        result_a = dg.materialize(
            [asset_a],
            resources=resources,
            instance=instance,
        )
        assert result_a.success

        # Verify only US data is in the store so far
        with mx.MetaxyConfig.get().get_store("dev") as store:
            data_after_a = store.read_metadata(shared_feature).collect()
            assert len(data_after_a) == 2
            assert set(data_after_a.to_native()["region"].to_list()) == {"us"}

        # Materialize asset_b (EU region)
        result_b = dg.materialize(
            [asset_b],
            resources=resources,
            instance=instance,
        )
        assert result_b.success

        # Verify both US and EU data are now in the store
        with mx.MetaxyConfig.get().get_store("dev") as store:
            all_data = store.read_metadata(shared_feature).collect()
            assert len(all_data) == 5  # 2 US + 3 EU
            assert set(all_data.to_native()["region"].to_list()) == {"us", "eu"}
            assert set(all_data.to_native()["id"].to_list()) == {
                "us_1",
                "us_2",
                "eu_1",
                "eu_2",
                "eu_3",
            }


class TestMultiAssetPartitionObservation:
    """Test observation with generate_observe_results for partitioned multi-assets."""

    def test_asset_c_observes_all_records(
        self,
        shared_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test Asset C with no metaxy/partition observes all records from A and B."""

        # First materialize asset_a (US region)
        @dg.asset(
            metadata={
                "metaxy/feature": "features/shared",
                "metaxy/partition": {"region": "us"},
            },
            io_manager_key="metaxy_io_manager",
        )
        def asset_a():
            return pl.DataFrame(
                {
                    "id": ["us_1", "us_2"],
                    "region": ["us", "us"],
                    "metaxy_provenance_by_field": [
                        {"value": "v1", "region": "fixed"},
                        {"value": "v2", "region": "fixed"},
                    ],
                }
            )

        # Then materialize asset_b (EU region)
        @dg.asset(
            metadata={
                "metaxy/feature": "features/shared",
                "metaxy/partition": {"region": "eu"},
            },
            io_manager_key="metaxy_io_manager",
        )
        def asset_b():
            return pl.DataFrame(
                {
                    "id": ["eu_1", "eu_2", "eu_3"],
                    "region": ["eu", "eu", "eu"],
                    "metaxy_provenance_by_field": [
                        {"value": "v3", "region": "fixed"},
                        {"value": "v4", "region": "fixed"},
                        {"value": "v5", "region": "fixed"},
                    ],
                }
            )

        dg.materialize([asset_a], resources=resources, instance=instance)
        dg.materialize([asset_b], resources=resources, instance=instance)

        # Create asset_c that observes the shared feature without partition metadata
        specs = [
            dg.AssetSpec(
                "asset_c",
                metadata={"metaxy/feature": "features/shared"},
                deps=[asset_a, asset_b],
            )
        ]

        @dg.multi_observable_source_asset(specs=specs)
        def asset_c(
            context: dg.AssetExecutionContext,
            store: mxd.MetaxyStoreFromConfigResource,
        ):
            yield from mxd.generate_observe_results(context, store, specs)

        # Run observation
        defs = dg.Definitions(
            assets=[asset_a, asset_b, asset_c],
            resources=resources,
        )
        job = defs.get_implicit_global_asset_job_def()
        result = job.execute_in_process(
            instance=instance, asset_selection=[dg.AssetKey("asset_c")]
        )
        assert result.success

        # Check observation event
        obs = instance.fetch_observations(dg.AssetKey("asset_c"), limit=1)
        assert len(obs.records) == 1

        metadata = obs.records[0].asset_observation.metadata  # pyright: ignore[reportOptionalMemberAccess]

        # Asset C should see all 5 records (2 US + 3 EU)
        assert metadata["dagster/row_count"].value == 5


class TestCompleteScenario:
    """Test the complete scenario: Assets A, B materialize; Asset C observes."""

    def test_complete_multi_asset_partition_flow(
        self,
        shared_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test complete flow: A materializes US, B materializes EU, C observes all."""

        # Define Asset A (US partition)
        @dg.asset(
            metadata={
                "metaxy/feature": "features/shared",
                "metaxy/partition": {"region": "us"},
            },
            io_manager_key="metaxy_io_manager",
        )
        def asset_a():
            """Materializes US region data."""
            return pl.DataFrame(
                {
                    "id": ["us_1", "us_2"],
                    "region": ["us", "us"],
                    "metaxy_provenance_by_field": [
                        {"value": "v1", "region": "fixed"},
                        {"value": "v2", "region": "fixed"},
                    ],
                }
            )

        # Define Asset B (EU partition)
        @dg.asset(
            metadata={
                "metaxy/feature": "features/shared",
                "metaxy/partition": {"region": "eu"},
            },
            io_manager_key="metaxy_io_manager",
        )
        def asset_b():
            """Materializes EU region data."""
            return pl.DataFrame(
                {
                    "id": ["eu_1", "eu_2", "eu_3"],
                    "region": ["eu", "eu", "eu"],
                    "metaxy_provenance_by_field": [
                        {"value": "v3", "region": "fixed"},
                        {"value": "v4", "region": "fixed"},
                        {"value": "v5", "region": "fixed"},
                    ],
                }
            )

        # Define Asset C (observes all data)
        specs_c = [
            dg.AssetSpec(
                "asset_c",
                metadata={"metaxy/feature": "features/shared"},
                deps=[asset_a, asset_b],
            )
        ]

        @dg.multi_observable_source_asset(specs=specs_c)
        def asset_c(
            context: dg.AssetExecutionContext,
            store: mxd.MetaxyStoreFromConfigResource,
        ):
            """Observes all shared feature data (no partition filter)."""
            yield from mxd.generate_observe_results(context, store, specs_c)

        # Step 1: Materialize Asset A
        result_a = dg.materialize([asset_a], resources=resources, instance=instance)
        assert result_a.success

        # Verify Asset A metadata
        mat_a = instance.fetch_materializations(dg.AssetKey("asset_a"), limit=1)
        assert len(mat_a.records) == 1
        metadata_a = mat_a.records[0].asset_materialization.metadata  # pyright: ignore[reportOptionalMemberAccess]
        assert metadata_a["dagster/row_count"].value == 2
        assert metadata_a["metaxy/materialized_in_run"].value == 2

        # Step 2: Materialize Asset B
        result_b = dg.materialize([asset_b], resources=resources, instance=instance)
        assert result_b.success

        # Verify Asset B metadata
        mat_b = instance.fetch_materializations(dg.AssetKey("asset_b"), limit=1)
        assert len(mat_b.records) == 1
        metadata_b = mat_b.records[0].asset_materialization.metadata  # pyright: ignore[reportOptionalMemberAccess]
        assert metadata_b["dagster/row_count"].value == 3
        assert metadata_b["metaxy/materialized_in_run"].value == 3

        # Step 3: Observe with Asset C
        defs = dg.Definitions(
            assets=[asset_a, asset_b, asset_c],
            resources=resources,
        )
        job = defs.get_implicit_global_asset_job_def()
        result_c = job.execute_in_process(
            instance=instance, asset_selection=[dg.AssetKey("asset_c")]
        )
        assert result_c.success

        # Verify Asset C observation sees all 5 records
        obs_c = instance.fetch_observations(dg.AssetKey("asset_c"), limit=1)
        assert len(obs_c.records) == 1
        metadata_c = obs_c.records[0].asset_observation.metadata  # pyright: ignore[reportOptionalMemberAccess]
        assert metadata_c["dagster/row_count"].value == 5

        # Verify the feature has all 5 records in the store
        with mx.MetaxyConfig.get().get_store("dev") as store:
            all_data = store.read_metadata(shared_feature).collect()
            assert len(all_data) == 5
            assert set(all_data.to_native()["region"].to_list()) == {"us", "eu"}
            assert set(all_data.to_native()["id"].to_list()) == {
                "us_1",
                "us_2",
                "eu_1",
                "eu_2",
                "eu_3",
            }


class TestMetaxyPartitionWithMultipleColumns:
    """Test metaxy/partition with multiple columns."""

    @pytest.fixture
    def multi_partition_feature(self) -> type[mx.BaseFeature]:
        """Create a feature with multiple partition columns."""
        spec = mx.FeatureSpec(
            key=["features", "multi_partition"],
            id_columns=["id"],
            fields=["region", "category", "value"],
        )

        class MultiPartitionFeature(mx.BaseFeature, spec=spec):
            id: str

        return MultiPartitionFeature

    def test_multi_column_partition(
        self,
        multi_partition_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that metaxy/partition can filter by multiple columns."""

        @dg.asset(
            metadata={
                "metaxy/feature": "features/multi_partition",
                "metaxy/partition": {"region": "us", "category": "premium"},
            },
            io_manager_key="metaxy_io_manager",
        )
        def asset_us_premium():
            return pl.DataFrame(
                {
                    "id": ["us_p_1", "us_p_2"],
                    "region": ["us", "us"],
                    "category": ["premium", "premium"],
                    "metaxy_provenance_by_field": [
                        {"value": "v1", "region": "fixed", "category": "fixed"},
                        {"value": "v2", "region": "fixed", "category": "fixed"},
                    ],
                }
            )

        @dg.asset(
            metadata={
                "metaxy/feature": "features/multi_partition",
                "metaxy/partition": {"region": "eu", "category": "basic"},
            },
            io_manager_key="metaxy_io_manager",
        )
        def asset_eu_basic():
            return pl.DataFrame(
                {
                    "id": ["eu_b_1", "eu_b_2", "eu_b_3"],
                    "region": ["eu", "eu", "eu"],
                    "category": ["basic", "basic", "basic"],
                    "metaxy_provenance_by_field": [
                        {"value": "v3", "region": "fixed", "category": "fixed"},
                        {"value": "v4", "region": "fixed", "category": "fixed"},
                        {"value": "v5", "region": "fixed", "category": "fixed"},
                    ],
                }
            )

        # Materialize both assets
        dg.materialize([asset_us_premium], resources=resources, instance=instance)
        dg.materialize([asset_eu_basic], resources=resources, instance=instance)

        # Verify each asset sees only its partition
        mat_us = instance.fetch_materializations(
            dg.AssetKey("asset_us_premium"), limit=1
        )
        mat_eu = instance.fetch_materializations(dg.AssetKey("asset_eu_basic"), limit=1)

        assert (
            mat_us.records[0]
            .asset_materialization.metadata["dagster/row_count"]  # pyright: ignore[reportOptionalMemberAccess]
            .value
            == 2
        )
        assert (
            mat_eu.records[0]
            .asset_materialization.metadata["dagster/row_count"]  # pyright: ignore[reportOptionalMemberAccess]
            .value
            == 3
        )

        # Verify both partitions are in the store
        with mx.MetaxyConfig.get().get_store("dev") as store:
            all_data = store.read_metadata(multi_partition_feature).collect()
            assert len(all_data) == 5
