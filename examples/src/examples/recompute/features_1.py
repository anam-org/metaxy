"""Feature definitions - Version 1.

This represents the initial feature implementation.
"""

from metaxy import (
    Feature,
    FeatureDep,
    FeatureKey,
    FeatureSpec,
    FieldDep,
    FieldKey,
    FieldSpec,
)


class ParentFeature(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["examples", "parent"]),
        deps=None,
        fields=[
            FieldSpec(
                key=FieldKey(["embeddings"]),
                code_version=1,  # Initial version
            ),
        ],
    ),
):
    """Parent feature that generates embeddings from raw data."""

    pass


class ChildFeature(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["examples", "child"]),
        deps=[FeatureDep(key=ParentFeature.spec().key)],
        fields=[
            FieldSpec(
                key=FieldKey(["predictions"]),
                code_version=1,
                deps=[
                    FieldDep(
                        feature_key=ParentFeature.spec().key,
                        fields=[FieldKey(["embeddings"])],
                    )
                ],
            ),
        ],
    ),
):
    """Child feature that uses parent embeddings to generate predictions."""

    pass
