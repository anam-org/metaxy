"""Tests for dagster utils."""

import dagster as dg
import pytest

import metaxy as mx
from metaxy.ext.dagster.constants import (
    DAGSTER_METAXY_FEATURE_METADATA_KEY,
    DAGSTER_METAXY_KIND,
    DAGSTER_METAXY_METADATA_METADATA_KEY,
)
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
    """Create a downstream feature with dependency."""
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
def feature_with_custom_key() -> type[mx.BaseFeature]:
    """Create a feature with custom dagster asset key."""
    spec = mx.FeatureSpec(
        key=["test", "custom"],
        id_columns=["id"],
        fields=["data"],
        metadata={
            DAGSTER_METAXY_METADATA_METADATA_KEY: ["custom", "dagster", "key"],
        },
    )

    class CustomKeyFeature(mx.BaseFeature, spec=spec):
        id: str

    return CustomKeyFeature


class TestBuildAssetSpec:
    """Tests for build_asset_spec function."""

    def test_build_asset_spec_from_string_key(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test building asset spec from string feature key."""
        spec = build_asset_spec("test/upstream")

        assert spec.key == dg.AssetKey(["test", "upstream"])
        assert spec.metadata[DAGSTER_METAXY_FEATURE_METADATA_KEY] == "test/upstream"
        assert DAGSTER_METAXY_KIND in spec.kinds

    def test_build_asset_spec_from_list_key(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test building asset spec from list feature key."""
        spec = build_asset_spec(["test", "upstream"])

        assert spec.key == dg.AssetKey(["test", "upstream"])
        assert spec.metadata[DAGSTER_METAXY_FEATURE_METADATA_KEY] == "test/upstream"

    def test_build_asset_spec_from_feature_class(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test building asset spec from feature class."""
        spec = build_asset_spec(upstream_feature)

        assert spec.key == dg.AssetKey(["test", "upstream"])
        assert spec.metadata[DAGSTER_METAXY_FEATURE_METADATA_KEY] == "test/upstream"

    def test_build_asset_spec_with_key_prefix(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that key_prefix is prepended to asset key."""
        spec = build_asset_spec("test/upstream", key_prefix=["prefix", "path"])

        assert spec.key == dg.AssetKey(["prefix", "path", "test", "upstream"])

    def test_build_asset_spec_includes_deps(
        self,
        upstream_feature: type[mx.BaseFeature],
        downstream_feature: type[mx.BaseFeature],
    ):
        """Test that upstream dependencies are included."""
        spec = build_asset_spec("test/downstream")

        dep_keys = {dep.asset_key for dep in spec.deps}
        assert dg.AssetKey(["test", "upstream"]) in dep_keys

    def test_build_asset_spec_deps_use_key_prefix(
        self,
        upstream_feature: type[mx.BaseFeature],
        downstream_feature: type[mx.BaseFeature],
    ):
        """Test that deps also use key_prefix."""
        spec = build_asset_spec("test/downstream", key_prefix=["prefix"])

        dep_keys = {dep.asset_key for dep in spec.deps}
        assert dg.AssetKey(["prefix", "test", "upstream"]) in dep_keys

    def test_build_asset_spec_exclude_kind(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that metaxy kind can be excluded."""
        spec = build_asset_spec("test/upstream", include_kind=False)

        assert DAGSTER_METAXY_KIND not in spec.kinds

    def test_build_asset_spec_uses_custom_dagster_key(
        self, feature_with_custom_key: type[mx.BaseFeature]
    ):
        """Test that custom dagster key from metadata is used."""
        spec = build_asset_spec("test/custom")

        assert spec.key == dg.AssetKey(["custom", "dagster", "key"])
        # But metaxy/feature metadata still uses original key
        assert spec.metadata[DAGSTER_METAXY_FEATURE_METADATA_KEY] == "test/custom"

    def test_build_asset_spec_no_deps_for_root_feature(
        self, upstream_feature: type[mx.BaseFeature]
    ):
        """Test that root feature has no deps."""
        spec = build_asset_spec("test/upstream")

        assert len(spec.deps) == 0  # pyright: ignore

    def test_build_asset_spec_includes_feature_metadata(
        self, feature_with_custom_key: type[mx.BaseFeature]
    ):
        """Test that feature spec metadata is included in asset spec."""
        spec = build_asset_spec("test/custom")

        # Should include the custom dagster key metadata from feature spec
        assert DAGSTER_METAXY_METADATA_METADATA_KEY in spec.metadata
        assert spec.metadata[DAGSTER_METAXY_METADATA_METADATA_KEY] == [
            "custom",
            "dagster",
            "key",
        ]
