"""Feature definitions for recompute example."""

from metaxy import (
    BaseFeature,
    FeatureSpec,
    FieldSpec,
)


class ParentFeature(
    BaseFeature,
    spec=FeatureSpec(
        key="examples/parent",
        fields=[
            FieldSpec(
                key="embeddings",
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
        key="examples/child",
        deps=[ParentFeature],
        fields=["predictions"],
        id_columns=("sample_uid",),
    ),
):
    """Child feature that uses parent embeddings to generate predictions."""

    pass
