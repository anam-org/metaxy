"""Feature definitions - Version 2.

ChildFeature has updated code_version to demonstrate migration.
ParentFeature remains unchanged.
"""

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
                code_version="1",  # Unchanged
            ),
        ],
        id_columns=("sample_uid",),
    ),
):
    """Parent feature (unchanged)."""

    pass


class ChildFeature(
    BaseFeature,
    spec=FeatureSpec(
        key=FeatureKey(["examples", "child"]),
        deps=[FeatureDep(feature=ParentFeature.spec().key)],
        fields=[
            FieldSpec(
                key=FieldKey(["predictions"]),
                code_version="2",  # ⚠️ CHANGED: Code refactor (same output)
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
    """Child feature with refactored code (same algorithm/output)."""

    pass
