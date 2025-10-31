"""Feature definitions - Version 1.

Initial feature implementation.
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
                code_version=1,
            ),
        ],
    ),
):
    """Parent feature generating embeddings."""

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
    """Child feature using parent embeddings."""

    pass
