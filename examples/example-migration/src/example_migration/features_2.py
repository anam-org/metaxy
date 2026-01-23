"""Feature definitions - Version 2.

ChildFeature has updated code_version to demonstrate migration.
ParentFeature remains unchanged.
"""

import metaxy as mx


class ParentFeature(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key=mx.FeatureKey(["examples", "parent"]),
        fields=[
            mx.FieldSpec(
                key=mx.FieldKey(["embeddings"]),
                code_version="1",  # Unchanged
            ),
        ],
        id_columns=("sample_uid",),
    ),
):
    """Parent feature (unchanged)."""

    pass


class ChildFeature(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key=mx.FeatureKey(["examples", "child"]),
        deps=[mx.FeatureDep(feature=ParentFeature.spec().key)],
        fields=[
            mx.FieldSpec(
                key=mx.FieldKey(["predictions"]),
                code_version="2",  # ⚠️ CHANGED: Code refactor (same output)
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
    """Child feature with refactored code (same algorithm/output)."""

    pass
