"""Feature definitions for recompute example."""

from metaxy import (
    BaseFeature,
    FeatureDep,
    FeatureKey,
    FeatureSpec,
    FieldDep,
    FieldKey,
    FieldSpec,
)


class ParentFeature(
    BaseFeature,
    spec=FeatureSpec(
        key=FeatureKey(["examples", "parent"]),
        fields=[
            FieldSpec(
                key=FieldKey(["embeddings"]),
                code_version="1",
            ),
        ],
        id_columns=("sample_uid",),
    ),
):
    """Parent feature that generates embeddings from raw data."""

    pass


class ChildFeature(
    BaseFeature,
    spec=FeatureSpec(
        key=FeatureKey(["examples", "child"]),
        deps=[FeatureDep(feature=ParentFeature.spec().key)],
        fields=[
            FieldSpec(
                key=FieldKey(["predictions"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature=ParentFeature.spec().key,
                        fields=[FieldKey(["embeddings"])],
                    )
                ],
            ),
        ],
        id_columns=("sample_uid",),
    ),
):
    """Child feature that uses parent embeddings to generate predictions."""

    pass
