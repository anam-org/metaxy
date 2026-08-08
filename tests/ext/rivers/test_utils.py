"""Tests for metaxy.ext.rivers.utils."""

import metaxy as mx
from metaxy.ext.rivers.utils import build_feature_info_metadata


def test_build_feature_info_metadata(metaxy_config: mx.MetaxyConfig):
    """Test that build_feature_info_metadata returns the expected structure."""
    spec = mx.FeatureSpec(
        key=["test", "info_metadata"],
        id_columns=["id"],
        fields=["value"],
    )

    class InfoMetadataFeature(mx.BaseFeature, spec=spec):
        id: str
        value: int

    info = build_feature_info_metadata("test/info_metadata")

    assert "feature" in info
    assert info["feature"]["project"] == "test"
    assert info["feature"]["spec"]["id_columns"] == ["id"]
    assert "version" in info["feature"]

    assert "metaxy" in info
    assert info["metaxy"]["version"] == mx.__version__