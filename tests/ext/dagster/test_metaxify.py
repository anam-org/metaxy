"""Tests for the @metaxify decorator."""

from typing import Any

import dagster as dg
import polars as pl
import pytest

import metaxy as mx
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_KIND,
    DAGSTER_METAXY_METADATA_METADATA_KEY,
)
from metaxy.ext.dagster.metaxify import metaxify
from metaxy.ext.dagster.utils import build_asset_spec


@pytest.fixture
def upstream_feature() -> type[mx.BaseFeature]:
    """Create an upstream feature."""
    spec = mx.FeatureSpec(
        key=["test", "upstream"],
        id_columns=["id"],
        fields=["value"],
    )

    class UpstreamFeature(mx.BaseFeature, spec=spec):
        id: str

    return UpstreamFeature


@pytest.fixture
def downstream_feature(
    upstream_feature: type[mx.BaseFeature],
) -> type[mx.BaseFeature]:
    """Create a downstream feature that depends on upstream."""
    spec = mx.FeatureSpec(
        key=["test", "downstream"],
        id_columns=["id"],
        fields=["result"],
        deps=[mx.FeatureDep(feature=upstream_feature)],
    )

    class DownstreamFeature(mx.BaseFeature, spec=spec):
        id: str

    return DownstreamFeature


@pytest.fixture
def feature_with_dagster_metadata() -> type[mx.BaseFeature]:
    """Create a feature with custom dagster asset key in metadata."""
    spec = mx.FeatureSpec(
        key=["test", "custom_key"],
        id_columns=["id"],
        fields=["data"],
        metadata={
            DAGSTER_METAXY_METADATA_METADATA_KEY: ["custom", "asset", "key"],
        },
    )

    class CustomKeyFeature(mx.BaseFeature, spec=spec):
        id: str

    return CustomKeyFeature


class TestMetaxifyBasic:
    """Test basic @metaxify functionality."""

    def test_metaxify_injects_metaxy_kind(self, upstream_feature: type[mx.BaseFeature]):
        """Test that metaxify injects 'metaxy' kind into asset."""

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def my_asset():
            pass

        # Check that metaxy kind was injected
        asset_spec = list(my_asset.specs)[0]
        assert DAGSTER_METAXY_KIND in asset_spec.kinds

    def test_metaxify_skips_kind_injection_when_disabled(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that kind injection can be disabled."""

        @metaxify(inject_metaxy_kind=False)
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        assert DAGSTER_METAXY_KIND not in asset_spec.kinds

    def test_metaxify_skips_kind_when_already_3_kinds(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that kind injection is skipped when asset already has 3 kinds."""

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/upstream"},
            kinds={"python", "sql", "dbt"},
        )
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        # Should have original 3 kinds, not metaxy
        assert len(asset_spec.kinds) == 3
        assert DAGSTER_METAXY_KIND not in asset_spec.kinds

    def test_metaxify_injects_feature_metadata(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that metaxify injects feature spec metadata."""

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        # Feature spec metadata should be injected
        assert asset_spec.metadata is not None


class TestMetaxifyAssetKeys:
    """Test asset key handling in @metaxify."""

    def test_metaxify_preserves_asset_key_by_default(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that asset key is preserved when inherit_feature_key_as_asset_key is False."""

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def my_asset():
            pass

        # Asset key should remain unchanged (original dagster key)
        assert dg.AssetKey(["my_asset"]) in my_asset.keys

    def test_metaxify_uses_feature_key_when_inherit_enabled(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that feature key is used as asset key when inherit_feature_key_as_asset_key is True."""

        @metaxify(inherit_feature_key_as_asset_key=True)
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def my_asset():
            pass

        # Asset key should be the feature key
        assert dg.AssetKey(["test", "upstream"]) in my_asset.keys

    def test_metaxify_uses_custom_key_from_feature_spec(
        self, feature_with_dagster_metadata: type[mx.BaseFeature]
    ):
        """Test that metaxy/metadata in feature spec overrides the asset key."""

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/custom_key"})
        def my_asset():
            pass

        # Asset key should use the custom key from feature spec's metaxy/metadata
        assert dg.AssetKey(["custom", "asset", "key"]) in my_asset.keys

    def test_metaxify_custom_key_overrides_inherit(
        self, feature_with_dagster_metadata: type[mx.BaseFeature]
    ):
        """Test that metaxy/metadata takes precedence over inherit_feature_key_as_asset_key."""

        @metaxify(inherit_feature_key_as_asset_key=True)
        @dg.asset(metadata={"metaxy/feature": "test/custom_key"})
        def my_asset():
            pass

        # Asset key should be custom key (metaxy/metadata takes precedence)
        assert dg.AssetKey(["custom", "asset", "key"]) in my_asset.keys


class TestMetaxifyDeps:
    """Test dependency injection in @metaxify."""

    def test_metaxify_injects_upstream_deps_with_feature_keys(
        self,
        upstream_feature: type[mx.BaseFeature],
        downstream_feature: type[mx.BaseFeature],
    ):
        """Test that upstream feature deps are injected using feature keys when inherit is enabled."""

        @metaxify(inherit_feature_key_as_asset_key=True)
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def upstream_asset():
            pass

        @metaxify(inherit_feature_key_as_asset_key=True)
        @dg.asset(metadata={"metaxy/feature": "test/downstream"})
        def downstream_asset():
            pass

        # Downstream asset should have upstream feature key as a dependency
        downstream_spec = list(downstream_asset.specs)[0]
        dep_keys = {dep.asset_key for dep in downstream_spec.deps}
        assert dg.AssetKey(["test", "upstream"]) in dep_keys

    def test_metaxify_deps_use_feature_keys_when_inherit_enabled(
        self,
        upstream_feature: type[mx.BaseFeature],
        downstream_feature: type[mx.BaseFeature],
    ):
        """Test that deps use feature keys when inherit_feature_key_as_asset_key is True."""

        @metaxify(inherit_feature_key_as_asset_key=True)
        @dg.asset(metadata={"metaxy/feature": "test/downstream"})
        def downstream_asset():
            pass

        downstream_spec = list(downstream_asset.specs)[0]
        dep_keys = {dep.asset_key for dep in downstream_spec.deps}
        # Dep key should use feature key
        assert dg.AssetKey(["test", "upstream"]) in dep_keys


class TestMetaxifyMixedDeps:
    """Test @metaxify with assets that have both Metaxy and non-Metaxy parent assets."""

    def test_metaxify_preserves_existing_dagster_deps(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that existing Dagster deps are preserved when metaxify adds Metaxy deps."""

        @dg.asset
        def regular_parent():
            pass

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/upstream"},
            deps=[regular_parent],
        )
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        dep_keys = {dep.asset_key for dep in asset_spec.deps}

        # Should have the regular parent as a dependency
        assert dg.AssetKey(["regular_parent"]) in dep_keys

    def test_metaxify_combines_metaxy_and_dagster_deps(
        self,
        upstream_feature: type[mx.BaseFeature],
        downstream_feature: type[mx.BaseFeature],
    ):
        """Test that Metaxy deps are added alongside existing Dagster deps."""

        @dg.asset
        def data_source():
            pass

        @dg.asset
        def another_source():
            pass

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/downstream"},
            deps=[data_source, another_source],
        )
        def my_downstream():
            pass

        asset_spec = list(my_downstream.specs)[0]
        dep_keys = {dep.asset_key for dep in asset_spec.deps}

        # Should have both regular Dagster deps
        assert dg.AssetKey(["data_source"]) in dep_keys
        assert dg.AssetKey(["another_source"]) in dep_keys
        # And the Metaxy upstream dep (from feature spec, using feature key)
        assert dg.AssetKey(["test", "upstream"]) in dep_keys

    def test_metaxify_with_mixed_metaxy_and_regular_assets_in_graph(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test a graph with both metaxified and regular assets."""

        @dg.asset
        def raw_data():
            return "raw"

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/upstream"},
            deps=[raw_data],
        )
        def metaxy_asset():
            pass

        @dg.asset(deps=[metaxy_asset])
        def consumer_asset():
            pass

        # metaxy_asset should have raw_data as dep and metaxy kind
        metaxy_spec = list(metaxy_asset.specs)[0]
        metaxy_dep_keys = {dep.asset_key for dep in metaxy_spec.deps}
        assert dg.AssetKey(["raw_data"]) in metaxy_dep_keys
        assert DAGSTER_METAXY_KIND in metaxy_spec.kinds

        # consumer_asset should depend on the original metaxy_asset key (not renamed)
        consumer_spec = list(consumer_asset.specs)[0]
        consumer_dep_keys = {dep.asset_key for dep in consumer_spec.deps}
        assert dg.AssetKey(["metaxy_asset"]) in consumer_dep_keys


class TestMetaxifyNoOp:
    """Test that @metaxify is a no-op for assets without metaxy metadata."""

    def test_metaxify_ignores_assets_without_metaxy_metadata(self):
        """Test that assets without metaxy/feature metadata are unchanged."""

        @metaxify()
        @dg.asset
        def regular_asset():
            pass

        # Asset should keep its original key
        assert dg.AssetKey(["regular_asset"]) in regular_asset.keys
        # Should not have metaxy kind
        asset_spec = list(regular_asset.specs)[0]
        assert DAGSTER_METAXY_KIND not in asset_spec.kinds


class TestMetaxifyMaterialization:
    """Test @metaxify with actual asset materialization."""

    def test_metaxify_asset_materializes_successfully(
        self,
        upstream_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that a metaxified asset can be materialized."""

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/upstream"},
            io_manager_key="metaxy_io_manager",
        )
        def upstream_asset():
            return pl.DataFrame(
                {
                    "id": ["1", "2"],
                    "metaxy_provenance_by_field": [
                        {"value": "v1"},
                        {"value": "v1"},
                    ],
                }
            )

        result = dg.materialize(
            [upstream_asset],
            resources=resources,
            instance=instance,
        )

        assert result.success

        # Verify metaxy metadata was logged
        event = result.get_asset_materialization_events()[0]
        metadata = event.step_materialization_data.materialization.metadata
        assert "metaxy/feature" in metadata
        assert "metaxy/materialized_in_run" in metadata
        assert metadata["metaxy/materialized_in_run"].value == 2

    def test_metaxify_upstream_downstream_chain(
        self,
        upstream_feature: type[mx.BaseFeature],
        downstream_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test materializing a chain of metaxified assets with dependencies."""

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/upstream"},
            io_manager_key="metaxy_io_manager",
        )
        def upstream_asset():
            return pl.DataFrame(
                {
                    "id": ["a", "b", "c"],
                    "metaxy_provenance_by_field": [
                        {"value": "h1"},
                        {"value": "h2"},
                        {"value": "h3"},
                    ],
                }
            )

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/downstream"},
            io_manager_key="metaxy_io_manager",
        )
        def downstream_asset(store: dg.ResourceParam[mx.MetadataStore]):
            # Read upstream data and verify it exists
            with store:
                upstream_data = store.read_metadata(upstream_feature).collect()
                assert len(upstream_data) == 3
            return pl.DataFrame(
                {
                    "id": ["x", "y"],
                    "metaxy_provenance_by_field": [
                        {"result": "r1"},
                        {"result": "r2"},
                    ],
                }
            )

        # Materialize upstream first
        result1 = dg.materialize(
            [upstream_asset],
            resources=resources,
            instance=instance,
        )
        assert result1.success

        # Materialize downstream - it should be able to read upstream data
        result2 = dg.materialize(
            [downstream_asset],
            resources=resources,
            instance=instance,
            selection=[downstream_asset],
        )
        assert result2.success

        # Verify downstream materialization metadata
        event = result2.get_asset_materialization_events()[0]
        metadata = event.step_materialization_data.materialization.metadata
        assert metadata["metaxy/materialized_in_run"].value == 2

    def test_metaxify_loads_upstream_via_store_resource(
        self,
        upstream_feature: type[mx.BaseFeature],
        downstream_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that downstream asset can read upstream data via the store resource."""
        captured_data = {}

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/upstream"},
            io_manager_key="metaxy_io_manager",
        )
        def upstream_asset():
            return pl.DataFrame(
                {
                    "id": ["x", "y", "z"],
                    "metaxy_provenance_by_field": [
                        {"value": "v1"},
                        {"value": "v2"},
                        {"value": "v3"},
                    ],
                }
            )

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/downstream"},
            io_manager_key="metaxy_io_manager",
            deps=[upstream_asset],
        )
        def downstream_asset(store: dg.ResourceParam[mx.MetadataStore]):
            # Read upstream data via the store resource
            with store:
                upstream_data = store.read_metadata(upstream_feature).collect()
                captured_data["rows"] = len(upstream_data)
                captured_data["ids"] = upstream_data.to_native()["id"].to_list()
            return pl.DataFrame(
                {
                    "id": ["result"],
                    "metaxy_provenance_by_field": [{"result": "done"}],
                }
            )

        # Materialize upstream first
        result1 = dg.materialize(
            [upstream_asset],
            resources=resources,
            instance=instance,
        )
        assert result1.success

        # Materialize downstream
        result2 = dg.materialize(
            [upstream_asset, downstream_asset],
            resources=resources,
            instance=instance,
            selection=[downstream_asset],
        )
        assert result2.success

        # Verify upstream data was read
        assert captured_data["rows"] == 3
        assert set(captured_data["ids"]) == {"x", "y", "z"}

    def test_metaxify_loads_upstream_via_build_asset_spec(
        self,
        upstream_feature: type[mx.BaseFeature],
        downstream_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that build_asset_spec can be used to reference external Metaxy features."""
        captured_data = {}

        # Build asset spec - this represents an external feature not produced by a local asset
        external_spec = build_asset_spec("test/upstream").with_io_manager_key(
            "metaxy_io_manager"
        )

        # First, write data directly to the store (simulating external feature population)
        with mx.MetaxyConfig.get().get_store("dev") as store:
            store.write_metadata(
                feature=upstream_feature,
                df=pl.DataFrame(
                    {
                        "id": ["a", "b"],
                        "metaxy_provenance_by_field": [
                            {"value": "v1"},
                            {"value": "v2"},
                        ],
                    }
                ),
            )

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/downstream"},
            io_manager_key="metaxy_io_manager",
            ins={
                # Reference the external feature using build_asset_spec key
                "upstream_data": dg.AssetIn(key=external_spec.key)
            },
        )
        def downstream_asset(upstream_data):
            collected = upstream_data.collect()
            captured_data["rows"] = len(collected)
            captured_data["ids"] = collected.to_native()["id"].to_list()
            return pl.DataFrame(
                {
                    "id": ["result"],
                    "metaxy_provenance_by_field": [{"result": "done"}],
                }
            )

        # Materialize downstream - pass external_spec so Dagster picks up its metadata
        result = dg.materialize(
            [external_spec, downstream_asset],
            resources=resources,
            instance=instance,
            selection=[downstream_asset],
        )
        assert result.success

        # Verify upstream data was loaded via IO Manager
        assert captured_data["rows"] == 2
        assert set(captured_data["ids"]) == {"a", "b"}

    def test_metaxify_with_regular_dagster_deps_materializes(
        self,
        upstream_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test metaxified asset with regular Dagster dependencies."""
        captured_data = {}

        @dg.asset
        def raw_data_source():
            return {"source": "raw"}

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/upstream"},
            io_manager_key="metaxy_io_manager",
            deps=[raw_data_source],
        )
        def metaxy_asset():
            captured_data["executed"] = True
            return pl.DataFrame(
                {
                    "id": ["1"],
                    "metaxy_provenance_by_field": [{"value": "v1"}],
                }
            )

        # Materialize both assets
        result = dg.materialize(
            [raw_data_source, metaxy_asset],
            resources=resources,
            instance=instance,
        )

        assert result.success
        assert captured_data.get("executed") is True

        # Verify the metaxy asset was materialized with correct key
        events = result.get_asset_materialization_events()
        asset_keys = {
            e.step_materialization_data.materialization.asset_key for e in events
        }
        # metaxy_asset key should be preserved (not replaced with feature key)
        assert dg.AssetKey(["metaxy_asset"]) in asset_keys

    def test_metaxify_preserves_asset_metadata_after_materialization(
        self,
        upstream_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that feature spec metadata is preserved after materialization."""

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/upstream"},
            io_manager_key="metaxy_io_manager",
        )
        def my_asset():
            return pl.DataFrame(
                {
                    "id": ["1"],
                    "metaxy_provenance_by_field": [{"value": "v1"}],
                }
            )

        result = dg.materialize(
            [my_asset],
            resources=resources,
            instance=instance,
        )

        assert result.success

        # Check that the metaxy kind was applied
        asset_spec = list(my_asset.specs)[0]
        assert DAGSTER_METAXY_KIND in asset_spec.kinds
