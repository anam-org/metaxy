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
    DAGSTER_METAXY_PROJECT_TAG_KEY,
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


class TestMetaxifyTags:
    """Test tag injection in @metaxify."""

    def test_metaxify_injects_feature_tag(self, upstream_feature: type[mx.BaseFeature]):
        """Test that metaxify injects metaxy/feature tag with feature key table_name."""

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        assert DAGSTER_METAXY_FEATURE_METADATA_KEY in asset_spec.tags
        # Tag value uses table_name format (__ separator) for Dagster compatibility
        assert asset_spec.tags[DAGSTER_METAXY_FEATURE_METADATA_KEY] == "test__upstream"

    def test_metaxify_injects_project_tag(self, upstream_feature: type[mx.BaseFeature]):
        """Test that metaxify injects metaxy/project tag with project name."""

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        assert DAGSTER_METAXY_PROJECT_TAG_KEY in asset_spec.tags
        assert (
            asset_spec.tags[DAGSTER_METAXY_PROJECT_TAG_KEY]
            == mx.MetaxyConfig.get().project
        )

    def test_metaxify_preserves_existing_tags(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that metaxify preserves existing asset tags."""

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/upstream"},
            tags={"custom_tag": "custom_value"},
        )
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        # Original tag preserved
        assert asset_spec.tags["custom_tag"] == "custom_value"
        # Metaxy tags injected
        assert DAGSTER_METAXY_FEATURE_METADATA_KEY in asset_spec.tags
        assert DAGSTER_METAXY_PROJECT_TAG_KEY in asset_spec.tags

    def test_metaxify_asset_spec_injects_tags(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that metaxify injects tags into AssetSpec."""
        spec = dg.AssetSpec(
            key="my_spec",
            metadata={"metaxy/feature": "test/upstream"},
        )

        result = metaxify()(spec)

        assert DAGSTER_METAXY_FEATURE_METADATA_KEY in result.tags
        # Tag value uses table_name format (__ separator) for Dagster compatibility
        assert result.tags[DAGSTER_METAXY_FEATURE_METADATA_KEY] == "test__upstream"
        assert DAGSTER_METAXY_PROJECT_TAG_KEY in result.tags


class TestMetaxifyAssetKeys:
    """Test asset key handling in @metaxify."""

    def test_metaxify_uses_feature_key_by_default(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that feature key is used as asset key by default."""

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def my_asset():
            pass

        # Asset key should be the feature key (default behavior)
        assert dg.AssetKey(["test", "upstream"]) in my_asset.keys

    def test_metaxify_preserves_asset_key_when_inherit_disabled(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that asset key is preserved when inherit_feature_key_as_asset_key is False."""

        @metaxify(inherit_feature_key_as_asset_key=False)
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def my_asset():
            pass

        # Asset key should remain unchanged (original dagster key)
        assert dg.AssetKey(["my_asset"]) in my_asset.keys

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


class TestMetaxifyKeyOverride:
    """Test explicit key and key_prefix parameters in @metaxify."""

    def test_metaxify_key_overrides_everything(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that explicit key parameter overrides all other key resolution."""

        @metaxify(key=["custom", "explicit", "key"])
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def my_asset():
            pass

        # Explicit key should override feature key
        assert dg.AssetKey(["custom", "explicit", "key"]) in my_asset.keys

    def test_metaxify_key_overrides_feature_spec_custom_key(
        self, feature_with_dagster_metadata: type[mx.BaseFeature]
    ):
        """Test that explicit key overrides even dagster/attributes.asset_key from feature spec."""

        @metaxify(key=["override", "key"])
        @dg.asset(metadata={"metaxy/feature": "test/custom_key"})
        def my_asset():
            pass

        # Explicit key should override feature spec's custom key
        assert dg.AssetKey(["override", "key"]) in my_asset.keys

    def test_metaxify_key_prefix_prepends_to_feature_key(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that key_prefix is prepended to the resolved feature key."""

        @metaxify(key_prefix=["prefix", "namespace"])
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def my_asset():
            pass

        # Key should be prefix + feature key
        assert dg.AssetKey(["prefix", "namespace", "test", "upstream"]) in my_asset.keys

    def test_metaxify_key_prefix_applies_to_deps(
        self,
        upstream_feature: type[mx.BaseFeature],
        downstream_feature: type[mx.BaseFeature],
    ):
        """Test that key_prefix is also applied to upstream dependency keys."""

        @metaxify(key_prefix=["prefix"])
        @dg.asset(metadata={"metaxy/feature": "test/downstream"})
        def downstream_asset():
            pass

        downstream_spec = list(downstream_asset.specs)[0]
        dep_keys = {dep.asset_key for dep in downstream_spec.deps}

        # Upstream dep should also have the prefix
        assert dg.AssetKey(["prefix", "test", "upstream"]) in dep_keys

    def test_metaxify_key_and_key_prefix_mutually_exclusive(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that key and key_prefix cannot be used together."""
        with pytest.raises(ValueError) as exc_info:
            metaxify(key=["explicit"], key_prefix=["prefix"])

        assert "Cannot specify both `key` and `key_prefix`" in str(exc_info.value)

    def test_metaxify_key_with_multi_asset_raises_error(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that explicit key cannot be used with multi-asset producing multiple outputs."""

        @dg.multi_asset(
            specs=[
                dg.AssetSpec("output_a", metadata={"metaxy/feature": "test/upstream"}),
                dg.AssetSpec("output_b"),
            ]
        )
        def my_multi_asset():
            pass

        with pytest.raises(ValueError) as exc_info:
            metaxify(key=["explicit", "key"])(my_multi_asset)

        assert "Cannot use `key` argument with multi-asset" in str(exc_info.value)

    def test_metaxify_key_prefix_works_with_multi_asset(
        self,
        upstream_feature: type[mx.BaseFeature],
    ):
        """Test that key_prefix works with multi-asset definitions."""

        @pytest.fixture
        def feature_b() -> type[mx.BaseFeature]:
            spec = mx.FeatureSpec(
                key=["test", "multi_b"],
                id_columns=["id"],
                fields=["value"],
            )

            class FeatureB(mx.BaseFeature, spec=spec):
                id: str

            return FeatureB

        # Create another feature for multi-asset
        spec_b = mx.FeatureSpec(
            key=["test", "multi_b"],
            id_columns=["id"],
            fields=["value"],
        )

        class FeatureB(mx.BaseFeature, spec=spec_b):
            id: str

        @metaxify(key_prefix=["prefix"])
        @dg.multi_asset(
            specs=[
                dg.AssetSpec("output_a", metadata={"metaxy/feature": "test/upstream"}),
                dg.AssetSpec("output_b", metadata={"metaxy/feature": "test/multi_b"}),
            ]
        )
        def my_multi_asset():
            pass

        # Both outputs should have the prefix
        assert dg.AssetKey(["prefix", "test", "upstream"]) in my_multi_asset.keys
        assert dg.AssetKey(["prefix", "test", "multi_b"]) in my_multi_asset.keys

    def test_metaxify_asset_spec_with_key(self, upstream_feature: type[mx.BaseFeature]):
        """Test that explicit key works with AssetSpec."""
        spec = dg.AssetSpec(
            key="my_spec",
            metadata={"metaxy/feature": "test/upstream"},
        )

        result = metaxify(key=["explicit", "key"])(spec)

        assert result.key == dg.AssetKey(["explicit", "key"])

    def test_metaxify_asset_spec_with_key_prefix(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that key_prefix works with AssetSpec."""
        spec = dg.AssetSpec(
            key="my_spec",
            metadata={"metaxy/feature": "test/upstream"},
        )

        result = metaxify(key_prefix=["prefix"])(spec)

        assert result.key == dg.AssetKey(["prefix", "test", "upstream"])

    def test_metaxify_key_with_string_coercion(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that key accepts string and coerces to AssetKey."""

        @metaxify(key="single_key")
        @dg.asset(metadata={"metaxy/feature": "test/upstream"})
        def my_asset():
            pass

        assert dg.AssetKey(["single_key"]) in my_asset.keys


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

        # metaxy_asset key should be the feature key
        assert dg.AssetKey(["test", "upstream"]) in metaxy_asset.keys

        # consumer_asset should depend on the feature key (metaxy_asset's transformed key)
        consumer_spec = list(consumer_asset.specs)[0]
        consumer_dep_keys = {dep.asset_key for dep in consumer_spec.deps}
        assert dg.AssetKey(["test", "upstream"]) in consumer_dep_keys


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
        # Key should be feature key (inherit_feature_key_as_asset_key=True by default)
        assert result.key == dg.AssetKey(["test", "upstream"])

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

        # Verify the metaxy asset was materialized with feature key
        events = result.get_asset_materialization_events()
        asset_keys = {
            e.step_materialization_data.materialization.asset_key for e in events
        }
        # metaxy_asset key should be the feature key (default behavior)
        assert dg.AssetKey(["test", "upstream"]) in asset_keys

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


class TestMetaxifyMultiAsset:
    """Test @metaxify with multi-asset definitions."""

    @pytest.fixture
    def feature_a(self) -> type[mx.BaseFeature]:
        """Create feature A for multi-asset tests."""
        spec = mx.FeatureSpec(
            key=["test", "multi", "a"],
            id_columns=["id"],
            fields=["value_a"],
        )

        class FeatureA(mx.BaseFeature, spec=spec):
            id: str

        return FeatureA

    @pytest.fixture
    def feature_b(self) -> type[mx.BaseFeature]:
        """Create feature B for multi-asset tests."""
        spec = mx.FeatureSpec(
            key=["test", "multi", "b"],
            id_columns=["id"],
            fields=["value_b"],
        )

        class FeatureB(mx.BaseFeature, spec=spec):
            id: str

        return FeatureB

    def test_metaxify_raises_error_with_feature_arg_on_multi_asset(
        self,
        feature_a: type[mx.BaseFeature],
    ):
        """Test that metaxify raises error when feature arg is used with multi-asset."""

        @dg.multi_asset(
            specs=[
                dg.AssetSpec("output_a"),
                dg.AssetSpec("output_b"),
            ]
        )
        def my_multi_asset():
            pass

        with pytest.raises(ValueError) as exc_info:
            metaxify(feature=feature_a)(my_multi_asset)

        assert "Cannot use `feature` argument with multi-asset" in str(exc_info.value)
        assert "my_multi_asset" in str(exc_info.value)
        assert "2 outputs" in str(exc_info.value)

    def test_metaxify_multi_asset_with_metadata_on_specs(
        self,
        feature_a: type[mx.BaseFeature],
        feature_b: type[mx.BaseFeature],
    ):
        """Test that metaxify works with multi-asset when metadata is on specs."""

        @metaxify()
        @dg.multi_asset(
            specs=[
                dg.AssetSpec("output_a", metadata={"metaxy/feature": "test/multi/a"}),
                dg.AssetSpec("output_b", metadata={"metaxy/feature": "test/multi/b"}),
            ]
        )
        def my_multi_asset():
            pass

        specs_by_key = my_multi_asset.specs_by_key
        assert len(specs_by_key) == 2

        # Keys should be feature keys (default behavior)
        spec_a = specs_by_key[dg.AssetKey(["test", "multi", "a"])]
        spec_b = specs_by_key[dg.AssetKey(["test", "multi", "b"])]

        # Both should have metaxy kind injected
        assert DAGSTER_METAXY_KIND in spec_a.kinds
        assert DAGSTER_METAXY_KIND in spec_b.kinds

        # Both should have metaxy metadata
        assert DAGSTER_METAXY_METADATA_METADATA_KEY in spec_a.metadata
        assert DAGSTER_METAXY_METADATA_METADATA_KEY in spec_b.metadata

    def test_metaxify_multi_asset_partial_metadata(
        self,
        feature_a: type[mx.BaseFeature],
    ):
        """Test that metaxify handles multi-asset where only some specs have metaxy metadata."""

        @metaxify()
        @dg.multi_asset(
            specs=[
                dg.AssetSpec("output_a", metadata={"metaxy/feature": "test/multi/a"}),
                dg.AssetSpec("output_b"),  # No metaxy metadata
            ]
        )
        def my_multi_asset():
            pass

        specs_by_key = my_multi_asset.specs_by_key

        # spec_a should have its key changed to feature key
        spec_a = specs_by_key[dg.AssetKey(["test", "multi", "a"])]
        # spec_b should keep its original key (no metaxy metadata)
        spec_b = specs_by_key[dg.AssetKey("output_b")]

        # Only spec_a should have metaxy enrichment
        assert DAGSTER_METAXY_KIND in spec_a.kinds
        assert DAGSTER_METAXY_METADATA_METADATA_KEY in spec_a.metadata

        # spec_b should be unchanged
        assert DAGSTER_METAXY_KIND not in spec_b.kinds
        assert DAGSTER_METAXY_METADATA_METADATA_KEY not in spec_b.metadata

    def test_metaxify_allows_feature_arg_on_single_output_multi_asset(
        self,
        feature_a: type[mx.BaseFeature],
    ):
        """Test that feature arg is allowed with multi_asset that has single output."""

        @dg.multi_asset(specs=[dg.AssetSpec("single_output")])
        def single_output_multi_asset():
            pass

        # Should not raise - single output is allowed
        result = metaxify(feature=feature_a)(single_output_multi_asset)

        spec = list(result.specs)[0]
        assert DAGSTER_METAXY_KIND in spec.kinds

    def test_metaxify_multi_asset_materializes_multiple_features(
        self,
        feature_a: type[mx.BaseFeature],
        feature_b: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that multi-asset with multiple metaxy features materializes correctly."""

        @metaxify()
        @dg.multi_asset(
            specs=[
                dg.AssetSpec(
                    "output_a",
                    metadata={"metaxy/feature": "test/multi/a"},
                ),
                dg.AssetSpec(
                    "output_b",
                    metadata={"metaxy/feature": "test/multi/b"},
                ),
            ]
        )
        def my_multi_asset(context: dg.AssetExecutionContext):
            # Use feature keys since inherit_feature_key_as_asset_key=True by default
            yield dg.MaterializeResult(
                asset_key=["test", "multi", "a"],
                metadata={"rows": 2},
            )
            yield dg.MaterializeResult(
                asset_key=["test", "multi", "b"],
                metadata={"rows": 3},
            )

        result = dg.materialize(
            [my_multi_asset],
            resources=resources,
            instance=instance,
        )
        assert result.success

        # Check both outputs were materialized with feature keys
        events = [
            e
            for e in result.all_events
            if e.event_type_value == "ASSET_MATERIALIZATION"
        ]
        assert len(events) == 2

        materialized_keys = {e.asset_key for e in events}
        assert dg.AssetKey(["test", "multi", "a"]) in materialized_keys
        assert dg.AssetKey(["test", "multi", "b"]) in materialized_keys


class TestMetaxifyWithInputDefinitions:
    """Test @metaxify with assets that have explicit input definitions (ins= parameter).

    This tests the fix for the Dagster bug where map_asset_specs/with_attributes fail
    on assets with InputDefinition objects because Dagster tries to call to_definition()
    on InputDefinition which doesn't have that method.
    """

    def test_metaxify_with_asset_in_parameter(
        self,
        upstream_feature: type[mx.BaseFeature],
    ):
        """Test that metaxify works on assets with ins= containing AssetIn objects."""
        # Create an external asset that we'll reference
        external_asset = dg.AssetSpec(key=["external", "data"])

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/upstream"},
            ins={
                "external_data": dg.AssetIn(key=external_asset.key),
            },
        )
        def my_asset(external_data):
            pass

        # Should not raise AttributeError: 'InputDefinition' object has no attribute 'to_definition'
        asset_spec = list(my_asset.specs)[0]
        assert DAGSTER_METAXY_KIND in asset_spec.kinds
        assert DAGSTER_METAXY_METADATA_METADATA_KEY in asset_spec.metadata

    def test_metaxify_with_multiple_asset_ins(
        self,
        upstream_feature: type[mx.BaseFeature],
    ):
        """Test metaxify with multiple AssetIn inputs."""
        external_a = dg.AssetSpec(key=["external", "a"])
        external_b = dg.AssetSpec(key=["external", "b"])

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/upstream"},
            ins={
                "input_a": dg.AssetIn(key=external_a.key),
                "input_b": dg.AssetIn(key=external_b.key),
            },
        )
        def my_asset(input_a, input_b):
            pass

        asset_spec = list(my_asset.specs)[0]
        assert DAGSTER_METAXY_KIND in asset_spec.kinds
        assert DAGSTER_METAXY_METADATA_METADATA_KEY in asset_spec.metadata

    def test_metaxify_with_asset_in_and_key_replacement(
        self,
        feature_with_dagster_metadata: type[mx.BaseFeature],
    ):
        """Test metaxify with AssetIn when asset key needs to be replaced."""
        external_asset = dg.AssetSpec(key=["external", "data"])

        # This feature has dagster/attributes.asset_key set, so key will be replaced
        @metaxify(feature=feature_with_dagster_metadata)
        @dg.asset(
            ins={
                "external_data": dg.AssetIn(key=external_asset.key),
            },
        )
        def original_key_asset(external_data):
            pass

        # Should work even with key replacement
        assert dg.AssetKey(["custom", "asset", "key"]) in original_key_asset.keys

    def test_metaxify_with_asset_in_materializes(
        self,
        upstream_feature: type[mx.BaseFeature],
        resources: dict[str, Any],
        instance: dg.DagsterInstance,
    ):
        """Test that metaxified asset with ins= can be materialized."""
        captured_data = {}

        @dg.asset(key=["source", "data"])
        def source_data():
            return {"value": 42}

        @metaxify()
        @dg.asset(
            metadata={"metaxy/feature": "test/upstream"},
            io_manager_key="metaxy_io_manager",
            ins={
                "source": dg.AssetIn(key=["source", "data"]),
            },
        )
        def my_asset(source):
            captured_data["source_value"] = source["value"]
            return pl.DataFrame(
                {
                    "id": ["1"],
                    "metaxy_provenance_by_field": [{"value": "v1"}],
                }
            )

        result = dg.materialize(
            [source_data, my_asset],
            resources=resources,
            instance=instance,
        )

        assert result.success
        assert captured_data["source_value"] == 42


class TestMetaxifyColumnSchema:
    """Test column schema injection in @metaxify."""

    def test_metaxify_injects_column_schema(self):
        """Test that metaxify injects column schema from Pydantic fields."""
        from pydantic import Field

        from metaxy.ext.dagster.constants import DAGSTER_COLUMN_SCHEMA_METADATA_KEY

        spec = mx.FeatureSpec(
            key=["test", "with_schema"],
            id_columns=["id"],
            fields=["value"],
        )

        class FeatureWithSchema(mx.BaseFeature, spec=spec):
            id: str = Field(description="Unique identifier")
            value: int = Field(description="Some integer value")
            name: str

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/with_schema"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        assert DAGSTER_COLUMN_SCHEMA_METADATA_KEY in asset_spec.metadata

        column_schema = asset_spec.metadata[DAGSTER_COLUMN_SCHEMA_METADATA_KEY]
        assert isinstance(column_schema, dg.TableSchema)

        columns_by_name = {col.name: col for col in column_schema.columns}

        assert "id" in columns_by_name
        assert columns_by_name["id"].description == "Unique identifier"
        assert columns_by_name["id"].type == "str"

        assert "value" in columns_by_name
        assert columns_by_name["value"].description == "Some integer value"
        assert columns_by_name["value"].type == "int"

        assert "name" in columns_by_name
        assert columns_by_name["name"].description is None
        assert columns_by_name["name"].type == "str"

    def test_metaxify_skips_column_schema_when_disabled(self):
        """Test that column schema injection can be disabled."""
        from metaxy.ext.dagster.constants import DAGSTER_COLUMN_SCHEMA_METADATA_KEY

        spec = mx.FeatureSpec(
            key=["test", "no_schema"],
            id_columns=["id"],
            fields=["value"],
        )

        class FeatureNoSchema(mx.BaseFeature, spec=spec):
            id: str
            value: int

        @metaxify(inject_column_schema=False)
        @dg.asset(metadata={"metaxy/feature": "test/no_schema"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        assert DAGSTER_COLUMN_SCHEMA_METADATA_KEY not in asset_spec.metadata

    def test_metaxify_asset_spec_injects_column_schema(self):
        """Test that metaxify injects column schema on AssetSpec."""
        from pydantic import Field

        from metaxy.ext.dagster.constants import DAGSTER_COLUMN_SCHEMA_METADATA_KEY

        spec = mx.FeatureSpec(
            key=["test", "spec_schema"],
            id_columns=["id"],
            fields=["data"],
        )

        class FeatureSpecSchema(mx.BaseFeature, spec=spec):
            id: str = Field(description="Record ID")
            data: float = Field(description="Data value")

        asset_spec = dg.AssetSpec(
            key="original_key",
            metadata={"metaxy/feature": "test/spec_schema"},
        )

        transformed_spec = metaxify()(asset_spec)

        assert DAGSTER_COLUMN_SCHEMA_METADATA_KEY in transformed_spec.metadata

        column_schema = transformed_spec.metadata[DAGSTER_COLUMN_SCHEMA_METADATA_KEY]
        columns_by_name = {col.name: col for col in column_schema.columns}

        assert "id" in columns_by_name
        assert columns_by_name["id"].description == "Record ID"
        assert columns_by_name["id"].type == "str"

        assert "data" in columns_by_name
        assert columns_by_name["data"].description == "Data value"
        assert columns_by_name["data"].type == "float"

    def test_metaxify_column_schema_with_complex_types(self):
        """Test column schema handles complex Pydantic types."""
        from pydantic import Field

        from metaxy.ext.dagster.constants import DAGSTER_COLUMN_SCHEMA_METADATA_KEY

        spec = mx.FeatureSpec(
            key=["test", "complex_types"],
            id_columns=["id"],
            fields=["default"],
        )

        class FeatureComplexTypes(mx.BaseFeature, spec=spec):
            id: str
            tags: list[str] = Field(description="List of tags")
            metadata: dict[str, Any] = Field(description="Arbitrary metadata")
            optional_value: int | None = Field(
                default=None, description="Optional integer"
            )

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/complex_types"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        column_schema = asset_spec.metadata[DAGSTER_COLUMN_SCHEMA_METADATA_KEY]
        columns_by_name = {col.name: col for col in column_schema.columns}

        # Check that complex types are stringified correctly
        assert "id" in columns_by_name
        assert columns_by_name["id"].type == "str"

        assert "tags" in columns_by_name
        assert columns_by_name["tags"].description == "List of tags"
        assert columns_by_name["tags"].type == "list[str]"

        assert "metadata" in columns_by_name
        assert columns_by_name["metadata"].description == "Arbitrary metadata"
        assert columns_by_name["metadata"].type == "dict[str, Any]"

        assert "optional_value" in columns_by_name
        assert columns_by_name["optional_value"].description == "Optional integer"
        assert columns_by_name["optional_value"].type == "int"


class TestMetaxifyColumnLineage:
    """Test column lineage injection in @metaxify."""

    def test_metaxify_injects_column_lineage_for_direct_passthrough(self):
        """Test column lineage injection when columns have the same name in both features."""
        from metaxy.ext.dagster.constants import DAGSTER_COLUMN_LINEAGE_METADATA_KEY

        upstream_spec = mx.FeatureSpec(
            key=["test", "lineage_upstream"],
            id_columns=["id"],
            fields=["value"],
        )

        class UpstreamFeature(mx.BaseFeature, spec=upstream_spec):
            id: str
            value: int
            name: str

        downstream_spec = mx.FeatureSpec(
            key=["test", "lineage_downstream"],
            id_columns=["id"],
            fields=["result"],
            deps=[mx.FeatureDep(feature=UpstreamFeature)],
        )

        class DownstreamFeature(mx.BaseFeature, spec=downstream_spec):
            id: str  # Same as upstream
            value: int  # Same as upstream
            result: float  # New column, no upstream

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/lineage_downstream"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        assert DAGSTER_COLUMN_LINEAGE_METADATA_KEY in asset_spec.metadata

        column_lineage = asset_spec.metadata[DAGSTER_COLUMN_LINEAGE_METADATA_KEY]
        assert isinstance(column_lineage, dg.TableColumnLineage)

        # ID column should have lineage
        assert "id" in column_lineage.deps_by_column
        id_deps = column_lineage.deps_by_column["id"]
        assert len(id_deps) == 1
        assert id_deps[0].column_name == "id"
        assert id_deps[0].asset_key == dg.AssetKey(["test", "lineage_upstream"])

        # Passthrough column should have lineage
        assert "value" in column_lineage.deps_by_column
        value_deps = column_lineage.deps_by_column["value"]
        assert len(value_deps) == 1
        assert value_deps[0].column_name == "value"
        assert value_deps[0].asset_key == dg.AssetKey(["test", "lineage_upstream"])

        # New column should not have lineage
        assert "result" not in column_lineage.deps_by_column

    def test_metaxify_column_lineage_with_rename(self):
        """Test column lineage respects FeatureDep.rename mappings."""
        from metaxy.ext.dagster.constants import DAGSTER_COLUMN_LINEAGE_METADATA_KEY

        upstream_spec = mx.FeatureSpec(
            key=["test", "rename_upstream"],
            id_columns=["id"],
            fields=["value"],
        )

        class UpstreamFeature(mx.BaseFeature, spec=upstream_spec):
            id: str
            old_name: str
            old_value: int

        downstream_spec = mx.FeatureSpec(
            key=["test", "rename_downstream"],
            id_columns=["id"],
            fields=["result"],
            deps=[
                mx.FeatureDep(
                    feature=UpstreamFeature,
                    rename={"old_name": "new_name", "old_value": "new_value"},
                )
            ],
        )

        class DownstreamFeature(mx.BaseFeature, spec=downstream_spec):
            id: str
            new_name: str  # Renamed from old_name
            new_value: int  # Renamed from old_value

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/rename_downstream"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        column_lineage = asset_spec.metadata[DAGSTER_COLUMN_LINEAGE_METADATA_KEY]

        # Renamed columns should trace back to original upstream names
        assert "new_name" in column_lineage.deps_by_column
        new_name_deps = column_lineage.deps_by_column["new_name"]
        assert len(new_name_deps) == 1
        assert new_name_deps[0].column_name == "old_name"
        assert new_name_deps[0].asset_key == dg.AssetKey(["test", "rename_upstream"])

        assert "new_value" in column_lineage.deps_by_column
        new_value_deps = column_lineage.deps_by_column["new_value"]
        assert len(new_value_deps) == 1
        assert new_value_deps[0].column_name == "old_value"

    def test_metaxify_column_lineage_with_aggregation_relationship(self):
        """Test column lineage respects aggregation lineage relationships."""
        from metaxy.ext.dagster.constants import DAGSTER_COLUMN_LINEAGE_METADATA_KEY

        upstream_spec = mx.FeatureSpec(
            key=["test", "agg_upstream"],
            id_columns=["user_id", "timestamp"],
            fields=["value"],
        )

        class UpstreamFeature(mx.BaseFeature, spec=upstream_spec):
            user_id: str
            timestamp: str
            value: int

        downstream_spec = mx.FeatureSpec(
            key=["test", "agg_downstream"],
            id_columns=["user_id"],  # Aggregated to user level
            fields=["total"],
            deps=[mx.FeatureDep(feature=UpstreamFeature)],
            lineage=mx.LineageRelationship.aggregation(on=["user_id"]),
        )

        class DownstreamFeature(mx.BaseFeature, spec=downstream_spec):
            user_id: str
            total: float

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/agg_downstream"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        column_lineage = asset_spec.metadata[DAGSTER_COLUMN_LINEAGE_METADATA_KEY]

        # Aggregation column should have lineage
        assert "user_id" in column_lineage.deps_by_column
        user_id_deps = column_lineage.deps_by_column["user_id"]
        assert len(user_id_deps) == 1
        assert user_id_deps[0].column_name == "user_id"

        # New computed column should not have lineage
        assert "total" not in column_lineage.deps_by_column

    def test_metaxify_column_lineage_with_expansion_relationship(self):
        """Test column lineage respects expansion lineage relationships."""
        from metaxy.ext.dagster.constants import DAGSTER_COLUMN_LINEAGE_METADATA_KEY

        upstream_spec = mx.FeatureSpec(
            key=["test", "exp_upstream"],
            id_columns=["doc_id"],
            fields=["content"],
        )

        class UpstreamFeature(mx.BaseFeature, spec=upstream_spec):
            doc_id: str
            content: str

        downstream_spec = mx.FeatureSpec(
            key=["test", "exp_downstream"],
            id_columns=["doc_id", "chunk_id"],  # Expanded with chunk_id
            fields=["chunk_text"],
            deps=[mx.FeatureDep(feature=UpstreamFeature)],
            lineage=mx.LineageRelationship.expansion(on=["doc_id"]),
        )

        class DownstreamFeature(mx.BaseFeature, spec=downstream_spec):
            doc_id: str
            chunk_id: str  # New ID column from expansion
            chunk_text: str

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/exp_downstream"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        column_lineage = asset_spec.metadata[DAGSTER_COLUMN_LINEAGE_METADATA_KEY]

        # Parent ID column should have lineage
        assert "doc_id" in column_lineage.deps_by_column
        doc_id_deps = column_lineage.deps_by_column["doc_id"]
        assert len(doc_id_deps) == 1
        assert doc_id_deps[0].column_name == "doc_id"

        # New chunk_id column should not have lineage (generated)
        assert "chunk_id" not in column_lineage.deps_by_column

    def test_metaxify_column_lineage_multiple_upstreams(self):
        """Test column lineage with multiple upstream dependencies."""
        from metaxy.ext.dagster.constants import DAGSTER_COLUMN_LINEAGE_METADATA_KEY

        upstream_a_spec = mx.FeatureSpec(
            key=["test", "multi_upstream_a"],
            id_columns=["id"],
            fields=["value_a"],
        )

        class UpstreamA(mx.BaseFeature, spec=upstream_a_spec):
            id: str
            value_a: int

        upstream_b_spec = mx.FeatureSpec(
            key=["test", "multi_upstream_b"],
            id_columns=["id"],
            fields=["value_b"],
        )

        class UpstreamB(mx.BaseFeature, spec=upstream_b_spec):
            id: str
            value_b: int

        downstream_spec = mx.FeatureSpec(
            key=["test", "multi_downstream"],
            id_columns=["id"],
            fields=["combined"],
            deps=[
                mx.FeatureDep(feature=UpstreamA),
                mx.FeatureDep(feature=UpstreamB),
            ],
        )

        class DownstreamFeature(mx.BaseFeature, spec=downstream_spec):
            id: str  # From both upstreams
            value_a: int  # From UpstreamA
            value_b: int  # From UpstreamB
            combined: float

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/multi_downstream"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        column_lineage = asset_spec.metadata[DAGSTER_COLUMN_LINEAGE_METADATA_KEY]

        # ID column should have lineage from both upstreams
        assert "id" in column_lineage.deps_by_column
        id_deps = column_lineage.deps_by_column["id"]
        assert len(id_deps) == 2  # From both upstreams
        asset_keys = {dep.asset_key for dep in id_deps}
        assert dg.AssetKey(["test", "multi_upstream_a"]) in asset_keys
        assert dg.AssetKey(["test", "multi_upstream_b"]) in asset_keys

        # value_a should only have lineage from UpstreamA
        assert "value_a" in column_lineage.deps_by_column
        value_a_deps = column_lineage.deps_by_column["value_a"]
        assert len(value_a_deps) == 1
        assert value_a_deps[0].asset_key == dg.AssetKey(["test", "multi_upstream_a"])

        # value_b should only have lineage from UpstreamB
        assert "value_b" in column_lineage.deps_by_column
        value_b_deps = column_lineage.deps_by_column["value_b"]
        assert len(value_b_deps) == 1
        assert value_b_deps[0].asset_key == dg.AssetKey(["test", "multi_upstream_b"])

    def test_metaxify_skips_column_lineage_when_disabled(self):
        """Test that column lineage injection can be disabled."""
        from metaxy.ext.dagster.constants import DAGSTER_COLUMN_LINEAGE_METADATA_KEY

        upstream_spec = mx.FeatureSpec(
            key=["test", "skip_lineage_upstream"],
            id_columns=["id"],
            fields=["value"],
        )

        class UpstreamFeature(mx.BaseFeature, spec=upstream_spec):
            id: str
            value: int

        downstream_spec = mx.FeatureSpec(
            key=["test", "skip_lineage_downstream"],
            id_columns=["id"],
            fields=["result"],
            deps=[mx.FeatureDep(feature=UpstreamFeature)],
        )

        class DownstreamFeature(mx.BaseFeature, spec=downstream_spec):
            id: str
            value: int

        @metaxify(inject_column_lineage=False)
        @dg.asset(metadata={"metaxy/feature": "test/skip_lineage_downstream"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        assert DAGSTER_COLUMN_LINEAGE_METADATA_KEY not in asset_spec.metadata

    def test_metaxify_no_column_lineage_without_deps(self):
        """Test that column lineage is not injected when there are no dependencies."""
        from metaxy.ext.dagster.constants import DAGSTER_COLUMN_LINEAGE_METADATA_KEY

        spec = mx.FeatureSpec(
            key=["test", "no_deps_lineage"],
            id_columns=["id"],
            fields=["value"],
        )

        class NoDepsFeature(mx.BaseFeature, spec=spec):
            id: str
            value: int

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/no_deps_lineage"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        # No column lineage because there are no dependencies
        assert DAGSTER_COLUMN_LINEAGE_METADATA_KEY not in asset_spec.metadata

    def test_metaxify_asset_spec_injects_column_lineage(self):
        """Test column lineage injection on AssetSpec."""
        from metaxy.ext.dagster.constants import DAGSTER_COLUMN_LINEAGE_METADATA_KEY

        upstream_spec = mx.FeatureSpec(
            key=["test", "spec_lineage_upstream"],
            id_columns=["id"],
            fields=["value"],
        )

        class UpstreamFeature(mx.BaseFeature, spec=upstream_spec):
            id: str
            value: int

        downstream_spec = mx.FeatureSpec(
            key=["test", "spec_lineage_downstream"],
            id_columns=["id"],
            fields=["result"],
            deps=[mx.FeatureDep(feature=UpstreamFeature)],
        )

        class DownstreamFeature(mx.BaseFeature, spec=downstream_spec):
            id: str
            value: int

        asset_spec = dg.AssetSpec(
            key="my_asset",
            metadata={"metaxy/feature": "test/spec_lineage_downstream"},
        )

        transformed_spec = metaxify()(asset_spec)

        assert DAGSTER_COLUMN_LINEAGE_METADATA_KEY in transformed_spec.metadata
        column_lineage = transformed_spec.metadata[DAGSTER_COLUMN_LINEAGE_METADATA_KEY]

        assert "id" in column_lineage.deps_by_column
        assert "value" in column_lineage.deps_by_column


class TestMetaxifySystemColumns:
    """Test system column injection in @metaxify for column schema and lineage."""

    def test_metaxify_injects_system_columns_in_schema(self):
        """Test that system columns are included in the column schema."""
        from metaxy.ext.dagster.constants import DAGSTER_COLUMN_SCHEMA_METADATA_KEY
        from metaxy.models.constants import (
            METAXY_CREATED_AT,
            METAXY_DATA_VERSION,
            METAXY_DATA_VERSION_BY_FIELD,
            METAXY_FEATURE_VERSION,
            METAXY_MATERIALIZATION_ID,
            METAXY_PROVENANCE,
            METAXY_PROVENANCE_BY_FIELD,
            METAXY_SNAPSHOT_VERSION,
        )

        spec = mx.FeatureSpec(
            key=["test", "system_schema"],
            id_columns=["id"],
            fields=["value"],
        )

        class FeatureWithSystemCols(mx.BaseFeature, spec=spec):
            id: str
            value: int

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/system_schema"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        assert DAGSTER_COLUMN_SCHEMA_METADATA_KEY in asset_spec.metadata

        column_schema = asset_spec.metadata[DAGSTER_COLUMN_SCHEMA_METADATA_KEY]
        columns_by_name = {col.name: col for col in column_schema.columns}

        # User columns should be present
        assert "id" in columns_by_name
        assert "value" in columns_by_name

        # All system columns should be present (inherited from BaseFeature)
        # metaxy_provenance_by_field is required, others are optional
        assert METAXY_PROVENANCE_BY_FIELD in columns_by_name
        assert columns_by_name[METAXY_PROVENANCE_BY_FIELD].type == "dict[str, str]"
        assert (
            "provenance"
            in columns_by_name[METAXY_PROVENANCE_BY_FIELD].description.lower()
        )

        assert METAXY_PROVENANCE in columns_by_name
        assert columns_by_name[METAXY_PROVENANCE].type == "str"

        assert METAXY_FEATURE_VERSION in columns_by_name
        assert columns_by_name[METAXY_FEATURE_VERSION].type == "str"

        assert METAXY_SNAPSHOT_VERSION in columns_by_name
        assert columns_by_name[METAXY_SNAPSHOT_VERSION].type == "str"

        assert METAXY_DATA_VERSION_BY_FIELD in columns_by_name
        assert columns_by_name[METAXY_DATA_VERSION_BY_FIELD].type == "dict[str, str]"

        assert METAXY_DATA_VERSION in columns_by_name
        assert columns_by_name[METAXY_DATA_VERSION].type == "str"

        assert METAXY_CREATED_AT in columns_by_name
        assert columns_by_name[METAXY_CREATED_AT].type == "datetime (UTC)"

        assert METAXY_MATERIALIZATION_ID in columns_by_name
        assert columns_by_name[METAXY_MATERIALIZATION_ID].type == "str"

    def test_metaxify_injects_system_column_lineage(self):
        """Test that system columns with lineage are tracked from upstream."""
        from metaxy.ext.dagster.constants import DAGSTER_COLUMN_LINEAGE_METADATA_KEY
        from metaxy.models.constants import (
            METAXY_PROVENANCE,
            METAXY_PROVENANCE_BY_FIELD,
        )

        upstream_spec = mx.FeatureSpec(
            key=["test", "syscol_lineage_upstream"],
            id_columns=["id"],
            fields=["value"],
        )

        class UpstreamFeature(mx.BaseFeature, spec=upstream_spec):
            id: str
            value: int

        downstream_spec = mx.FeatureSpec(
            key=["test", "syscol_lineage_downstream"],
            id_columns=["id"],
            fields=["result"],
            deps=[mx.FeatureDep(feature=UpstreamFeature)],
        )

        class DownstreamFeature(mx.BaseFeature, spec=downstream_spec):
            id: str

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/syscol_lineage_downstream"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        column_lineage = asset_spec.metadata[DAGSTER_COLUMN_LINEAGE_METADATA_KEY]

        # System columns with lineage should be present
        assert METAXY_PROVENANCE_BY_FIELD in column_lineage.deps_by_column
        prov_by_field_deps = column_lineage.deps_by_column[METAXY_PROVENANCE_BY_FIELD]
        assert len(prov_by_field_deps) == 1
        assert prov_by_field_deps[0].column_name == METAXY_PROVENANCE_BY_FIELD
        assert prov_by_field_deps[0].asset_key == dg.AssetKey(
            ["test", "syscol_lineage_upstream"]
        )

        assert METAXY_PROVENANCE in column_lineage.deps_by_column
        prov_deps = column_lineage.deps_by_column[METAXY_PROVENANCE]
        assert len(prov_deps) == 1
        assert prov_deps[0].column_name == METAXY_PROVENANCE
        assert prov_deps[0].asset_key == dg.AssetKey(
            ["test", "syscol_lineage_upstream"]
        )

    def test_metaxify_system_column_lineage_multiple_upstreams(self):
        """Test that system columns have lineage from all upstream dependencies."""
        from metaxy.ext.dagster.constants import DAGSTER_COLUMN_LINEAGE_METADATA_KEY
        from metaxy.models.constants import (
            METAXY_PROVENANCE,
            METAXY_PROVENANCE_BY_FIELD,
        )

        upstream_a_spec = mx.FeatureSpec(
            key=["test", "syscol_multi_upstream_a"],
            id_columns=["id"],
            fields=["value_a"],
        )

        class UpstreamA(mx.BaseFeature, spec=upstream_a_spec):
            id: str

        upstream_b_spec = mx.FeatureSpec(
            key=["test", "syscol_multi_upstream_b"],
            id_columns=["id"],
            fields=["value_b"],
        )

        class UpstreamB(mx.BaseFeature, spec=upstream_b_spec):
            id: str

        downstream_spec = mx.FeatureSpec(
            key=["test", "syscol_multi_downstream"],
            id_columns=["id"],
            fields=["combined"],
            deps=[
                mx.FeatureDep(feature=UpstreamA),
                mx.FeatureDep(feature=UpstreamB),
            ],
        )

        class DownstreamFeature(mx.BaseFeature, spec=downstream_spec):
            id: str

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/syscol_multi_downstream"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        column_lineage = asset_spec.metadata[DAGSTER_COLUMN_LINEAGE_METADATA_KEY]

        # System columns should have lineage from both upstreams
        assert METAXY_PROVENANCE_BY_FIELD in column_lineage.deps_by_column
        prov_by_field_deps = column_lineage.deps_by_column[METAXY_PROVENANCE_BY_FIELD]
        assert len(prov_by_field_deps) == 2
        asset_keys = {dep.asset_key for dep in prov_by_field_deps}
        assert dg.AssetKey(["test", "syscol_multi_upstream_a"]) in asset_keys
        assert dg.AssetKey(["test", "syscol_multi_upstream_b"]) in asset_keys

        assert METAXY_PROVENANCE in column_lineage.deps_by_column
        prov_deps = column_lineage.deps_by_column[METAXY_PROVENANCE]
        assert len(prov_deps) == 2

    def test_metaxify_no_system_column_lineage_without_deps(self):
        """Test that system columns don't have lineage when there are no upstream deps."""
        from metaxy.ext.dagster.constants import DAGSTER_COLUMN_LINEAGE_METADATA_KEY

        spec = mx.FeatureSpec(
            key=["test", "syscol_no_deps"],
            id_columns=["id"],
            fields=["value"],
        )

        class NoDepsFeature(mx.BaseFeature, spec=spec):
            id: str

        @metaxify()
        @dg.asset(metadata={"metaxy/feature": "test/syscol_no_deps"})
        def my_asset():
            pass

        asset_spec = list(my_asset.specs)[0]
        # No column lineage at all when there are no dependencies
        assert DAGSTER_COLUMN_LINEAGE_METADATA_KEY not in asset_spec.metadata
