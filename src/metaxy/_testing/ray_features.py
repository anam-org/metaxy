"""Feature definitions for Ray integration tests.

This module contains test features that are loaded via entrypoints
when testing Ray Data integration.
"""

from metaxy._testing.models import SampleFeatureSpec
from metaxy.models.feature import BaseFeature
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
    """Test feature for Ray actor tests."""

    __test__ = False  # Prevent pytest from collecting this as a test class

    sample_uid: str
    value: int


__all__ = ["RayTestFeature"]
