"""Test features for project detection."""

from metaxy import Feature, FeatureKey, FieldKey, FieldSpec
from metaxy._testing.models import SampleFeatureSpec


class TestFeature(
    Feature,
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
    Feature,
    spec=SampleFeatureSpec(
        key=FeatureKey(["test", "another"]),
        fields=[
            FieldSpec(key=FieldKey(["data"]), code_version="1"),
        ],
    ),
):
    """Another test feature."""

    pass
