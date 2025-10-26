"""Feature definitions - Version 2.

ParentFeature.code_version changed from 1 to 2.
This represents updating the embedding algorithm.
When parent changes, child must recompute even though its code is unchanged.
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
                code_version=2,  # ⚠️ CHANGED: Updated embedding algorithm
            ),
        ],
    ),
):
    """Parent feature with updated embedding algorithm."""

    pass


class ChildFeature(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["examples", "child"]),
        deps=[FeatureDep(key=ParentFeature.spec.key)],
        fields=[
            FieldSpec(
                key=FieldKey(["predictions"]),
                code_version=1,  # Unchanged - but will recompute due to parent change
                deps=[
                    FieldDep(
                        feature_key=ParentFeature.spec.key,
                        fields=[FieldKey(["embeddings"])],
                    )
                ],
            ),
        ],
    ),
):
    """Child feature (unchanged code, but will recompute due to parent change)."""

    pass
