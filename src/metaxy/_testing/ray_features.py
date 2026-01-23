"""Feature definitions for Ray integration tests.

This module contains test features that are loaded via entrypoints
when testing Ray Data integration.
"""

from metaxy._testing.models import SampleFeatureSpec
from metaxy.models.feature import BaseFeature
from metaxy.models.feature_spec import FeatureDep
from metaxy.models.field import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey


class RayTestFeature(
    BaseFeature,
    spec=SampleFeatureSpec(
        key=FeatureKey(["test", "ray_feature"]),
        fields=[
            FieldSpec(key=FieldKey(["value"]), code_version="1"),
        ],
    ),
):
    """Test feature for Ray actor tests (root feature)."""

    __test__ = False  # Prevent pytest from collecting this as a test class

    sample_uid: str
    value: int


class RayDerivedFeature(
    BaseFeature,
    spec=SampleFeatureSpec(
        key=FeatureKey(["test", "ray_derived"]),
        deps=[FeatureDep(feature=RayTestFeature)],
        fields=[
            FieldSpec(key=FieldKey(["derived_value"]), code_version="1"),
        ],
    ),
):
    """Test feature that depends on RayTestFeature (non-root feature)."""

    __test__ = False  # Prevent pytest from collecting this as a test class

    sample_uid: str
    derived_value: int


__all__ = ["RayTestFeature", "RayDerivedFeature"]
