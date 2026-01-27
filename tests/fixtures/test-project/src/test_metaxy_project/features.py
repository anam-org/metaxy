"""Test features for project detection."""

from metaxy import BaseFeature, FeatureKey, FeatureSpec, FieldKey, FieldSpec


class TestFeature(
    BaseFeature,
    spec=FeatureSpec(
        key=FeatureKey(["test", "feature"]),
        id_columns=["sample_uid"],
        fields=[
            FieldSpec(key=FieldKey(["value"]), code_version="1"),
        ],
    ),
):
    """A simple test feature."""

    sample_uid: str


class AnotherTestFeature(
    BaseFeature,
    spec=FeatureSpec(
        key=FeatureKey(["test", "another"]),
        id_columns=["sample_uid"],
        fields=[
            FieldSpec(key=FieldKey(["data"]), code_version="1"),
        ],
    ),
):
    """Another test feature."""

    sample_uid: str
