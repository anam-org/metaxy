"""Tests for metaxy.ext.rivers.metaxify."""

import metaxy as mx
from metaxy.ext.rivers.constants import (
    RIVERS_METAXY_FEATURE_METADATA_KEY,
    RIVERS_METAXY_INFO_METADATA_KEY,
)
from metaxy.ext.rivers.metaxify import metaxify


def test_metaxify_sets_asset_name_to_table_name(metaxy_config: mx.MetaxyConfig):
    """The asset name should always be the feature's table_name, regardless of the function name."""
    spec = mx.FeatureSpec(
        key=["test", "name_from_table"],
        id_columns=["id"],
        fields=["value"],
    )

    class NameFromTableFeature(mx.BaseFeature, spec=spec):
        id: str
        value: int

    @metaxify("test/name_from_table")
    def some_other_function_name():
        pass

    feature_key = mx.coerce_to_feature_key("test/name_from_table")
    expected_table_name = feature_key.table_name

    assert some_other_function_name.name == expected_table_name


def test_metaxify_injects_metadata(metaxy_config: mx.MetaxyConfig):
    """metaxify should inject the feature key and feature info into asset metadata."""
    spec = mx.FeatureSpec(
        key=["test", "metadata_injection"],
        id_columns=["id"],
        fields=["value"],
    )

    class MetadataInjectionFeature(mx.BaseFeature, spec=spec):
        id: str
        value: int

    @metaxify("test/metadata_injection")
    def my_asset():
        pass

    metadata = my_asset.metadata
    assert metadata is not None
    assert metadata[RIVERS_METAXY_FEATURE_METADATA_KEY] == "test/metadata_injection"
    assert RIVERS_METAXY_INFO_METADATA_KEY in metadata
    assert "metaxy/feature_code_version" in metadata


def test_metaxify_preserves_user_metadata(metaxy_config: mx.MetaxyConfig):
    """User-supplied metadata should be preserved alongside injected metaxy metadata."""
    spec = mx.FeatureSpec(
        key=["test", "user_metadata"],
        id_columns=["id"],
        fields=["value"],
    )

    class UserMetadataFeature(mx.BaseFeature, spec=spec):
        id: str
        value: int

    @metaxify("test/user_metadata", metadata={"custom_key": "custom_value"})
    def my_asset():
        pass

    metadata = my_asset.metadata
    assert metadata["custom_key"] == "custom_value"
    assert metadata[RIVERS_METAXY_FEATURE_METADATA_KEY] == "test/user_metadata"


def test_metaxify_injects_code_version(metaxy_config: mx.MetaxyConfig):
    """metaxify should inject a metaxy:<code_version> string into code_version."""
    spec = mx.FeatureSpec(
        key=["test", "code_version_injection"],
        id_columns=["id"],
        fields=["value"],
    )

    class CodeVersionFeature(mx.BaseFeature, spec=spec):
        id: str
        value: int

    @metaxify("test/code_version_injection")
    def my_asset():
        pass

    assert my_asset.code_version is not None
    assert my_asset.code_version.startswith("metaxy:")


def test_metaxify_disable_code_version_injection(metaxy_config: mx.MetaxyConfig):
    """When inject_code_version=False and no code_version is passed, code_version should be None."""
    spec = mx.FeatureSpec(
        key=["test", "no_code_version"],
        id_columns=["id"],
        fields=["value"],
    )

    class NoCodeVersionFeature(mx.BaseFeature, spec=spec):
        id: str
        value: int

    @metaxify("test/no_code_version", inject_code_version=False)
    def my_asset():
        pass

    assert my_asset.code_version is None


def test_metaxify_passes_through_asset_kwargs(metaxy_config: mx.MetaxyConfig):
    """Unrecognized asset_kwargs (e.g. tags, group) should be passed straight through to Asset(...)."""
    spec = mx.FeatureSpec(
        key=["test", "kwargs_passthrough"],
        id_columns=["id"],
        fields=["value"],
    )

    class KwargsPassthroughFeature(mx.BaseFeature, spec=spec):
        id: str
        value: int

    @metaxify("test/kwargs_passthrough", tags=["custom_tag"], group="custom_group")
    def my_asset():
        pass

    assert my_asset.tags == ["custom_tag"]
    assert my_asset.group == "custom_group"