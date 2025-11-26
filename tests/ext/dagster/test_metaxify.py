"""Tests for the @metaxify decorator."""

from typing import Any

import dagster as dg
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

import metaxy as mx
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_KIND,
    DAGSTER_METAXY_METADATA_METADATA_KEY,
    METAXY_DAGSTER_METADATA_KEY,
)
from metaxy.ext.dagster.metaxify import metaxify


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
            METAXY_DAGSTER_METADATA_KEY: {"asset_key": ["custom", "asset", "key"]},
        },
    )

    class CustomKeyFeature(mx.BaseFeature, spec=spec):
        id: str

    return CustomKeyFeature


@pytest.fixture
def feature_with_group_name() -> type[mx.BaseFeature]:
    """Create a feature with group_name in dagster/attributes."""
    spec = mx.FeatureSpec(
        key=["test", "grouped"],
        id_columns=["id"],
        fields=["data"],
        metadata={
            METAXY_DAGSTER_METADATA_KEY: {"group_name": "my_group"},
        },
    )

    class GroupedFeature(mx.BaseFeature, spec=spec):
        id: str

    return GroupedFeature


@pytest.fixture
def feature_with_multiple_attrs() -> type[mx.BaseFeature]:
    """Create a feature with multiple dagster attributes."""
    spec = mx.FeatureSpec(
        key=["test", "multi_attrs"],
        id_columns=["id"],
        fields=["data"],
        metadata={
            METAXY_DAGSTER_METADATA_KEY: {
                "asset_key": ["custom", "multi"],
                "group_name": "features",
                "owners": ["team:data", "user@example.com"],
            },
        },
    )

    class MultiAttrsFeature(mx.BaseFeature, spec=spec):
        id: str

    return MultiAttrsFeature


class TestMetaxifyBasic:
    """Test basic @metaxify functionality."""

    def test_metaxify_without_parentheses(self, upstream_feature: type[mx.BaseFeature]):
        """Test that @metaxify works without parentheses."""

        @metaxify
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def my_asset():
            pass

        # Should have metaxy kind injected
        asset_spec = list(my_asset.specs)[0]
        assert DAGSTER_METAXY_KIND in asset_spec.kinds

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
        """Test that metaxify injects feature spec metadata under DAGSTER_METAXY_METADATA_METADATA_KEY."""

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        # Feature spec metadata should be injected under the namespaced key
        assert DAGSTER_METAXY_METADATA_METADATA_KEY in asset_spec.metadata
        assert asset_spec.metadata[DAGSTER_METAXY_METADATA_METADATA_KEY] == {}

    def test_metaxify_injects_feature_metadata_with_content(
        self, feature_with_group_name: type[mx.BaseFeature]
    ):
        """Test that metaxify injects non-empty feature spec metadata."""

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/grouped"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        # Feature spec metadata should include the dagster/attributes
        assert DAGSTER_METAXY_METADATA_METADATA_KEY in asset_spec.metadata
        injected_metadata = asset_spec.metadata[DAGSTER_METAXY_METADATA_METADATA_KEY]
        assert METAXY_DAGSTER_METADATA_KEY in injected_metadata
        assert injected_metadata[METAXY_DAGSTER_METADATA_KEY] == {
            "group_name": "my_group"
        }


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


class TestMetaxifyDagsterAttributes:
    """Test dagster/attributes injection from feature spec metadata."""

    def test_metaxify_applies_group_name(
        self, feature_with_group_name: type[mx.BaseFeature]
    ):
        """Test that group_name from dagster/attributes is applied to the asset."""

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/grouped"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        assert asset_spec.group_name == "my_group"

    def test_metaxify_applies_multiple_attributes(
        self, feature_with_multiple_attrs: type[mx.BaseFeature]
    ):
        """Test that multiple dagster attributes are applied."""

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/multi_attrs"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        # asset_key is handled separately, so check the key
        assert dg.AssetKey(["custom", "multi"]) in my_asset.keys
        # Other attributes should be applied
        assert asset_spec.group_name == "features"
        assert asset_spec.owners == ["team:data", "user@example.com"]

    def test_metaxify_attributes_override_asset_attributes(
        self, feature_with_group_name: type[mx.BaseFeature]
    ):
        """Test that dagster/attributes override attributes set on the asset decorator."""

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/grouped"},
            group_name="original_group",
        )
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        # Feature spec's group_name should override the decorator's group_name
        assert asset_spec.group_name == "my_group"

    def test_metaxify_no_attributes_preserves_original(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that assets without dagster/attributes keep their original attributes."""

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/upstream"},
            group_name="my_original_group",
        )
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        # Original group_name should be preserved
        assert asset_spec.group_name == "my_original_group"


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


class TestMetaxifyAssetSpec:
    """Test @metaxify with raw AssetSpec objects."""

    def test_metaxify_transforms_asset_spec(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that metaxify can transform a raw AssetSpec."""
        spec = dg.AssetSpec(
            key="my_spec",
            metadata={"metaxy/feature": "test/upstream"},
        )

        result = metaxify()(spec)

        # Should return an AssetSpec
        assert isinstance(result, dg.AssetSpec)
        # Should have metaxy kind
        assert DAGSTER_METAXY_KIND in result.kinds
        # Key should be preserved (inherit_feature_key_as_asset_key=False by default)
        assert result.key == dg.AssetKey(["my_spec"])

    def test_metaxify_asset_spec_with_inherit_key(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that AssetSpec key is replaced when inherit_feature_key_as_asset_key=True."""
        spec = dg.AssetSpec(
            key="my_spec",
            metadata={"metaxy/feature": "test/upstream"},
        )

        result = metaxify(inherit_feature_key_as_asset_key=True)(spec)

        # Key should be the feature key
        assert result.key == dg.AssetKey(["test", "upstream"])

    def test_metaxify_asset_spec_with_custom_key(
        self, feature_with_dagster_metadata: type[mx.BaseFeature]
    ):
        """Test that AssetSpec uses custom key from dagster/attributes."""
        spec = dg.AssetSpec(
            key="my_spec",
            metadata={"metaxy/feature": "test/custom_key"},
        )

        result = metaxify()(spec)

        # Should use custom key from feature spec metadata
        assert result.key == dg.AssetKey(["custom", "asset", "key"])

    def test_metaxify_asset_spec_injects_deps(
        self,
        upstream_feature: type[mx.BaseFeature],
        downstream_feature: type[mx.BaseFeature],
    ):
        """Test that AssetSpec gets deps injected from feature spec."""
        spec = dg.AssetSpec(
            key="my_downstream",
            metadata={"metaxy/feature": "test/downstream"},
        )

        result = metaxify(inherit_feature_key_as_asset_key=True)(spec)

        dep_keys = {dep.asset_key for dep in result.deps}
        assert dg.AssetKey(["test", "upstream"]) in dep_keys

    def test_metaxify_asset_spec_applies_dagster_attrs(
        self, feature_with_group_name: type[mx.BaseFeature]
    ):
        """Test that AssetSpec gets dagster attributes from feature spec."""
        spec = dg.AssetSpec(
            key="my_spec",
            metadata={"metaxy/feature": "test/grouped"},
        )

        result = metaxify()(spec)

        assert result.group_name == "my_group"

    def test_metaxify_asset_spec_preserves_existing_deps(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that existing deps on AssetSpec are preserved."""
        spec = dg.AssetSpec(
            key="my_spec",
            metadata={"metaxy/feature": "test/upstream"},
            deps=[dg.AssetDep("existing_dep")],
        )

        result = metaxify()(spec)

        dep_keys = {dep.asset_key for dep in result.deps}
        assert dg.AssetKey(["existing_dep"]) in dep_keys

    def test_metaxify_asset_spec_no_op_without_feature(self):
        """Test that AssetSpec without metaxy/feature is returned unchanged."""
        spec = dg.AssetSpec(key="regular_spec")

        result = metaxify()(spec)

        assert result.key == dg.AssetKey(["regular_spec"])
        assert DAGSTER_METAXY_KIND not in result.kinds


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

    def test_metaxify_loads_upstream_via_external_asset_spec(
        self,
        upstream_feature: type[mx.BaseFeature],
        downstream_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that metaxify(AssetSpec) can be used to reference external Metaxy features."""
        captured_data = {}

        # Build asset spec using metaxify - represents an external feature not produced locally
        external_spec = metaxify(inherit_feature_key_as_asset_key=True)(
            dg.AssetSpec(
                key="external_upstream",
                metadata={DAGSTER_METAXY_FEATURE_METADATA_KEY: "test/upstream"},
            )
        ).with_io_manager_key("metaxy_io_manager")

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
                # Reference the external feature using metaxify'd AssetSpec key
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


class TestMetaxifyCodeVersion:
    """Test code version injection in @metaxify."""

    def test_metaxify_injects_code_version(
        self,
        upstream_feature: type[mx.BaseFeature],
        snapshot: SnapshotAssertion,
    ):
        """Test that metaxify injects the feature code version."""

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        assert asset_spec.code_version is not None
        assert asset_spec.code_version.startswith("metaxy:")
        assert asset_spec.code_version == snapshot

    def test_metaxify_appends_to_existing_code_version(
        self,
        upstream_feature: type[mx.BaseFeature],
        snapshot: SnapshotAssertion,
    ):
        """Test that metaxify appends to existing code version."""

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/upstream"},
            code_version="v1.0.0",
        )
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        assert asset_spec.code_version is not None
        assert asset_spec.code_version.startswith("v1.0.0,metaxy:")
        assert asset_spec.code_version == snapshot

    def test_metaxify_skips_code_version_when_disabled(
        self,
        upstream_feature: type[mx.BaseFeature],
    ):
        """Test that code version injection can be disabled."""

        @metaxify(inject_code_version=False)
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        assert asset_spec.code_version is None

    def test_metaxify_preserves_code_version_when_disabled(
        self,
        upstream_feature: type[mx.BaseFeature],
    ):
        """Test that existing code version is preserved when injection is disabled."""

        @metaxify(inject_code_version=False)
        @dg.asset(
            metadata={"metaxy/feature": "test/upstream"},
            code_version="original_version",
        )
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        assert asset_spec.code_version == "original_version"

    def test_metaxify_asset_spec_injects_code_version(
        self,
        upstream_feature: type[mx.BaseFeature],
        snapshot: SnapshotAssertion,
    ):
        """Test that metaxify injects code version into AssetSpec."""
        spec = dg.AssetSpec(
            key="my_spec",
            metadata={"metaxy/feature": "test/upstream"},
        )

        result = metaxify()(spec)

        assert result.code_version is not None
        assert result.code_version.startswith("metaxy:")
        assert result.code_version == snapshot


class TestMetaxifyDescription:
    """Test description injection in @metaxify."""

    @pytest.fixture
    def feature_with_docstring(self) -> type[mx.BaseFeature]:
        """Create a feature with a docstring."""
        spec = mx.FeatureSpec(
            key=["test", "documented"],
            id_columns=["id"],
            fields=["value"],
        )

        class DocumentedFeature(mx.BaseFeature, spec=spec):
            """This is the feature documentation.

            It describes what this feature does.
            """

            id: str

        return DocumentedFeature

    def test_metaxify_injects_description_from_docstring(
        self,
        feature_with_docstring: type[mx.BaseFeature],
        snapshot: SnapshotAssertion,
    ):
        """Test that metaxify injects description from feature docstring."""

        @metaxify(feature=feature_with_docstring)
        @dg.asset
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        assert asset_spec.description is not None
        assert "feature documentation" in asset_spec.description
        assert asset_spec.description == snapshot

    def test_metaxify_preserves_existing_description(
        self,
        feature_with_docstring: type[mx.BaseFeature],
    ):
        """Test that metaxify preserves existing asset description."""

        @metaxify(feature=feature_with_docstring)
        @dg.asset(description="My custom description")
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        assert asset_spec.description == "My custom description"

    def test_metaxify_no_description_without_docstring(
        self,
        upstream_feature: type[mx.BaseFeature],
    ):
        """Test that metaxify doesn't set description if feature has no docstring."""

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        # upstream_feature fixture has no docstring
        assert asset_spec.description is None

    def test_metaxify_skips_description_when_disabled(
        self,
        feature_with_docstring: type[mx.BaseFeature],
    ):
        """Test that description injection can be disabled."""

        @metaxify(feature=feature_with_docstring, set_description=False)
        @dg.asset
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        assert asset_spec.description is None

    def test_metaxify_asset_spec_injects_description(
        self,
        feature_with_docstring: type[mx.BaseFeature],
        snapshot: SnapshotAssertion,
    ):
        """Test that metaxify injects description into AssetSpec."""
        spec = dg.AssetSpec(key="my_spec")

        result = metaxify(feature=feature_with_docstring)(spec)

        assert result.description is not None
        assert "feature documentation" in result.description
        assert result.description == snapshot
