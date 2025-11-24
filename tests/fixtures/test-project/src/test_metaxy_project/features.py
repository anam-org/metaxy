"""Test features for project detection."""

from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
from metaxy._testing.models import SampleFeatureSpec


class TestFeature(
    BaseFeature,
    spec=SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[
            FieldSpec(key=FieldKey(["value"]), code_version="1"),
        ],
    ),
):
    """A simple test feature."""

    pass


class AnotherTestFeature(
    BaseFeature,
    spec=SampleFeatureSpec(
        key=FeatureKey(["test", "another"]),
        fields=[
            FieldSpec(key=FieldKey(["data"]), code_version="1"),
        ],
    ),
):
    """Another test feature."""

    pass
