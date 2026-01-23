"""Feature definitions for recompute example."""

# --8<-- [start:parent_feature]
import metaxy as mx


class ParentFeature(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="examples/parent",
        fields=[
            mx.FieldSpec(
                key="embeddings",
                code_version="1",
            ),
        ],
        id_columns=("sample_uid",),
    ),
):
    """Parent feature that generates embeddings from raw data."""

    pass


# --8<-- [end:parent_feature]


# --8<-- [start:child_feature]
class ChildFeature(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="examples/child",
        deps=[ParentFeature],
        fields=["predictions"],
        id_columns=("sample_uid",),
    ),
):
    """Child feature that uses parent embeddings to generate predictions."""

    pass


# --8<-- [end:child_feature]
