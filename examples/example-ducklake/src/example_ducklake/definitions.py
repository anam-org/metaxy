"""Feature definitions used by the DuckLake example."""

import metaxy as mx


class DuckLakeDemoFeature(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="examples/ducklake_demo",
        fields=["path"],
        id_columns=("sample_uid",),
    ),
):
    """Small feature used to show DuckLake-backed metadata persistence."""
