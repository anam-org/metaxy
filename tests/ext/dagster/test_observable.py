"""Tests for observable_metaxy_asset decorator."""

from typing import Any

import dagster as dg
import polars as pl
import pytest

import metaxy as mx
import metaxy.ext.dagster as mxd


@pytest.fixture
def observable_feature() -> type[mx.BaseFeature]:
    """Create a feature for observation tests."""
    spec = mx.FeatureSpec(
        key=["test", "observable"],
        id_columns=["id"],
        fields=["value"],
    )

    class ObservableFeature(mx.BaseFeature, spec=spec):
        """A feature that can be observed.

        This docstring should be used as the asset description.
        """

        id: str

    return ObservableFeature


def _run_observation(
    asset: dg.SourceAsset,
    resources: dict[str, Any],
) -> dg.ExecuteInProcessResult:
    """Run an observation on a source asset using Definitions."""
    defs = dg.Definitions(assets=[asset], resources=resources)
    job = defs.get_implicit_global_asset_job_def()
    return job.execute_in_process()


def _get_observations(
    result: dg.ExecuteInProcessResult,
) -> list[dg.AssetObservation]:
    """Get AssetObservation objects from a result."""
    observations: list[dg.AssetObservation] = []
    for event in result.all_events:
        if event.event_type_value == "ASSET_OBSERVATION":
            obs = getattr(event.event_specific_data, "asset_observation", None)
            if obs is not None:
                observations.append(obs)
    return observations


class TestObserveMetaxyAsset:
    """Test observable_metaxy_asset decorator."""

    def test_observe_feature_with_data(
        self,
        observable_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that observing a feature with data returns correct version and row count."""

        # First, write some data to the store
        @mxd.metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/observable"},
            io_manager_key="metaxy_io_manager",
        )
        def write_data():
            return pl.DataFrame(
                {
                    "id": ["1", "2", "3"],
                    "metaxy_provenance_by_field": [
                        {"value": "v1"},
                        {"value": "v2"},
                        {"value": "v3"},
                    ],
                }
            )

        # Materialize the data
        write_result = dg.materialize(
            [write_data],
            resources=resources,
            instance=instance,
        )
        assert write_result.success

        # Now create and run the observable asset
        @mxd.observable_metaxy_asset(feature=observable_feature, key="observable_asset")
        def observable_asset(context, store, lazy_df):
            pass

        result = _run_observation(observable_asset, resources)
        assert result.success

        observations = _get_observations(result)
        assert len(observations) == 1

        observation = observations[0]
        assert observation is not None
        assert observation.metadata["dagster/row_count"].value == 3

        # Data version should be set (not empty)
        assert observation.data_version is not None
        assert observation.data_version != "empty"

    def test_observe_detects_changes(
        self,
        observable_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that the data version changes when data is added."""

        # Write initial data
        @mxd.metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/observable"},
            io_manager_key="metaxy_io_manager",
        )
        def write_initial():
            return pl.DataFrame(
                {
                    "id": ["1"],
                    "metaxy_provenance_by_field": [{"value": "v1"}],
                }
            )

        dg.materialize(
            [write_initial],
            resources=resources,
            instance=instance,
        )

        # First observation
        @mxd.observable_metaxy_asset(feature=observable_feature, key="observable_asset")
        def observable_asset(context, store, lazy_df):
            pass

        result1 = _run_observation(observable_asset, resources)
        observations1 = _get_observations(result1)
        version1 = observations1[0].data_version

        # Write more data
        @mxd.metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/observable"},
            io_manager_key="metaxy_io_manager",
        )
        def write_more():
            return pl.DataFrame(
                {
                    "id": ["2"],
                    "metaxy_provenance_by_field": [{"value": "v2"}],
                }
            )

        dg.materialize(
            [write_more],
            resources=resources,
            instance=instance,
        )

        # Second observation
        result2 = _run_observation(observable_asset, resources)
        observations2 = _get_observations(result2)
        version2 = observations2[0].data_version

        # Versions should be different
        assert version1 != version2

    def test_observe_creates_source_asset(
        self,
        observable_feature: type[mx.BaseFeature],
    ):
        """Test that observable_metaxy_asset creates a SourceAsset."""

        @mxd.observable_metaxy_asset(feature=observable_feature)
        def my_observable(context, store, lazy_df):
            pass

        # Check it's a SourceAsset
        assert isinstance(my_observable, dg.SourceAsset)

        # Check that it's observable
        assert my_observable.is_observable

        # Check the key is derived from the feature key
        assert my_observable.key == dg.AssetKey(["test", "observable"])

    def test_observe_with_custom_metadata(
        self,
        observable_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that decorated function can add custom metadata via context."""

        # Write some data first
        @mxd.metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/observable"},
            io_manager_key="metaxy_io_manager",
        )
        def write_data():
            return pl.DataFrame(
                {
                    "id": ["1", "2"],
                    "metaxy_provenance_by_field": [
                        {"value": "v1"},
                        {"value": "v2"},
                    ],
                }
            )

        dg.materialize([write_data], resources=resources, instance=instance)

        # Create observable that returns custom metadata
        @mxd.observable_metaxy_asset(feature=observable_feature, key="observable_asset")
        def observable_asset(context, store, lazy_df):
            # Run aggregation in the database, collect only what we need
            ids = lazy_df.select("id").collect()["id"].to_list()
            return {"custom/ids": ids}

        result = _run_observation(observable_asset, resources)
        assert result.success

        observations = _get_observations(result)
        observation = observations[0]

        # Check custom metadata is present (sorted because order isn't guaranteed)
        assert "custom/ids" in observation.metadata
        assert sorted(observation.metadata["custom/ids"].value) == ["1", "2"]
