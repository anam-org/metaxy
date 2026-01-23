"""Feature definitions - Version 1.

Initial feature implementation.
"""

import metaxy as mx


class ParentFeature(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key=mx.FeatureKey(["examples", "parent"]),
        fields=[
            mx.FieldSpec(
                key=mx.FieldKey(["embeddings"]),
                code_version="1",
            ),
        ],
        id_columns=("sample_uid",),
    ),
):
    """Parent feature generating embeddings."""

    pass


class ChildFeature(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key=mx.FeatureKey(["examples", "child"]),
        deps=[mx.FeatureDep(feature=ParentFeature.spec().key)],
        fields=[
            mx.FieldSpec(
                key=mx.FieldKey(["predictions"]),
                code_version="1",
                deps=[
                    mx.FieldDep(
                        feature=ParentFeature.spec().key,
                        fields=[mx.FieldKey(["embeddings"])],
                    )
                ],
            ),
        ],
        id_columns=("sample_uid",),
    ),
):
    """Child feature using parent embeddings."""

    pass
