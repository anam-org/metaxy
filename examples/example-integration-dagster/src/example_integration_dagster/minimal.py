"""Minimal Dagster + Metaxy example using the @mxd.asset decorator.

Execute this example as outlined below:

Example
```bash
source .venv/bin/activate
cd example-integration-dagster/
dagster dev -f src/example_integration_dagster/minimal.py
```
"""

import dagster as dg
import narwhals as nw
import polars as pl

import metaxy as mx
import metaxy.ext.dagster as mxd
from metaxy.versioning.types import Increment

# Resources
store_resource = mxd.MetaxyMetadataStoreResource.from_config(store_name="dev")
io_manager = mxd.MetaxyIOManager.from_store(store_resource)


# Define Features
class RawFeature(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="demo/raw",
        id_columns=["id"],
        fields=[mx.FieldSpec(key="value", code_version="1")],
    ),
):
    id: str
    value: str


class CleanFeature(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="demo/clean",
        deps=[mx.FeatureDep(feature="demo/raw")],
        id_columns=["id"],
        fields=[mx.FieldSpec(key="clean_value", code_version="1")],
    ),
):
    id: str
    clean_value: str


# Define Assets using @mxd.asset decorator
@mxd.asset(feature=RawFeature)
def raw_feature(
    context: dg.AssetExecutionContext,
    store: mxd.MetaxyMetadataStoreResource,
) -> mx.FeatureSpec:
    """Root feature: generate samples with provenance and write metadata."""
    samples_df = pl.DataFrame(
        {
            "id": ["1"],
            "value": ["raw value"],
            "metaxy_provenance_by_field": [{"value": "hash_1"}],
        }
    )
    samples = nw.from_native(samples_df)

    with store:
        inc = store.resolve_update(RawFeature, samples=samples)
        if len(inc.added) > 0:
            store.write_metadata(RawFeature, inc.added)

    return RawFeature.spec()


@mxd.asset(feature=CleanFeature)
def clean_feature(
    context: dg.AssetExecutionContext,
    diff: Increment,  # Automatically injected by decorator for downstream features
    store: mxd.MetaxyMetadataStoreResource,
) -> mx.FeatureSpec:
    """Downstream feature: diff automatically provided by IOManager."""
    context.log.info(
        f"Processing {len(diff.added)} added, {len(diff.changed)} changed samples"
    )

    with store:
        if len(diff.added) > 0:
            # Use narwhals operations (works with any backend)
            cleaned = diff.added.with_columns(
                nw.col("value").str.replace("raw", "clean").alias("clean_value")
            )
            store.write_metadata(CleanFeature, cleaned)

    return CleanFeature.spec()


defs = dg.Definitions(
    assets=[raw_feature, clean_feature],
    resources={"store": store_resource, "io_manager": io_manager},
)
