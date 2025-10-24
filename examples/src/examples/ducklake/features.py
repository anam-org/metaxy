"""Feature definitions for DuckLake example."""

from metaxy import (
    Feature,
    FeatureDep,
    FeatureKey,
    FeatureSpec,
    FieldDep,
    FieldKey,
    FieldSpec,
)


class DuckLakeParentFeature(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["examples", "ducklake_parent"]),
        deps=None,
        fields=[
            FieldSpec(
                key=FieldKey(["ingested"]),
                code_version=1,
            ),
        ],
    ),
):
    """Upstream feature that simulates data ingested into DuckLake."""

    pass


class DuckLakeChildFeature(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["examples", "ducklake_child"]),
        deps=[FeatureDep(key=DuckLakeParentFeature.spec.key)],
        fields=[
            FieldSpec(
                key=FieldKey(["metrics"]),
                code_version=1,
                deps=[
                    FieldDep(
                        feature_key=DuckLakeParentFeature.spec.key,
                        fields=[FieldKey(["ingested"])],
                    )
                ],
            ),
        ],
    ),
):
    """Downstream feature that depends on DuckLake parent metadata."""

    pass
