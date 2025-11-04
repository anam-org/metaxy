"""Test features for project detection."""

from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec


class TestFeature(
    Feature,
    spec=FeatureSpec(
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
    spec=FeatureSpec(
        key=FeatureKey(["test", "another"]),
        fields=[
            FieldSpec(key=FieldKey(["data"]), code_version="1"),
        ],
    ),
):
    """Another test feature."""

    pass
