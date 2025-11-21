# Dagster Integration

The [Dagster](https://dagster.io/) integration lets Metaxy features participate in Dagster asset runs.
You getmetadata store resources, an IOManager that moves `Increment`s between assets, and helpers to turn features into Dagster assets.

## Installation

```bash
pip install "metaxy[dagster]"
```

## Quick Start

Minimal root + downstream wiring:

```python
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
    context.log.info("Generating raw samples...")
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
        context.log.info(
            f"RawFeature increment: {len(inc.added)} added, "
            f"{len(inc.changed)} changed, {len(inc.removed)} removed"
        )
        if len(inc.added) > 0:
            store.write_metadata(RawFeature, inc.added)

    return RawFeature.spec()


@mxd.asset(feature=CleanFeature)
def clean_feature(
    context: dg.AssetExecutionContext,
    diff: Increment,
    store: mxd.MetaxyMetadataStoreResource,
) -> mx.FeatureSpec:
    """Downstream feature: diff automatically provided by IOManager."""
    context.log.info(
        f"Processing {len(diff.added)} added, {len(diff.changed)} changed samples"
    )

    if len(diff.added) == 0:
        context.log.info("No added samples to clean")
        return CleanFeature.spec()

    with store:
        for chunk in mxd.iter_dataframe_with_progress(
            diff.added,
            chunk_size=100,
            desc="clean_feature",
            log_fn=context.log,
            log_level="debug",
            failed_count=0,
            echo_to_stderr=True,
        ):
            cleaned = chunk.with_columns(
                nw.col("value").str.replace("raw", "clean").alias("clean_value")
            )
            store.write_metadata(CleanFeature, cleaned)

    return CleanFeature.spec()


defs = dg.Definitions(
    assets=[raw_feature, clean_feature],
    resources={"store": store_resource, "io_manager": io_manager},
)
```

## Core Components

- [`MetaxyMetadataStoreResource`][metaxy.ext.dagster.MetaxyMetadataStoreResource]: load configured stores (with optional fallbacks).
  See class docstring for options.
- [`MetaxyIOManager`][metaxy.ext.dagster.MetaxyIOManager]: moves `Increment`s between assets; supports partition, sample, and key filtering.
  See the docstring for flags (`partition_key_column`, `sample_limit`, `sample_filter_expr`, `target_key_column`/`target_keys`).
- Helpers:
- `build_asset_spec_from_feature`: turns a feature into an `AssetSpec` and carries feature metadata (keys, versions, user metadata) into Dagster metadata.
- `build_asset_in_from_feature`: creates typed `AssetIn` mapping (optionally overriding which feature to resolve).
- Sampling helpers `sampling_config_schema` / `apply_sampling`: standard config + sampler you can add to any root asset for per-run random/key sampling.

## Execution patterns

Most code lives in the runnable examples and helper docstrings. Pick a pattern and start from the example file:

- Non-partitioned: process everything at once (default IOManager) — [examples/example-integration-dagster/non_partitioned.py](https://github.com/anam-org/metaxy/blob/main/examples/example-integration-dagster/src/example_integration_dagster/non_partitioned.py)
- Partitioned: one category per partition (each covering multiple videos and many rows); root filters manually, downstream filtered by IOManager — [examples/example-integration-dagster/partitioned.py](https://github.com/anam-org/metaxy/blob/main/examples/example-integration-dagster/src/example_integration_dagster/partitioned.py)
- Branch + subsample: read root from production fallback into a branch store, and choose sampling per run (random sample size or explicit keys) — [examples/example-integration-dagster/branch_subsampled.py](https://github.com/anam-org/metaxy/blob/main/examples/example-integration-dagster/src/example_integration_dagster/branch_subsampled.py)
- Shared sampling helper: use `sampling_config_schema()` + `apply_sampling()` (exported from `metaxy.ext.dagster`) to give any root asset an opt-in sampling config (keys vs random size) without custom wiring.

For targeted runs without partitions, set `target_key_column`/`target_keys` on `MetaxyIOManager` or via `AssetIn` metadata.
Partition labels alone do not filter data; root assets must filter samples, and downstream assets rely on the IOManager.

Notes on trade-offs:

- Non-partitioned is the common, simple path (one big diff), but can be heavy on memory/compute and needs tuning at scale.
- Partitioned parallel shines for isolating a single entity (better retries/observability per key, easier resource spikes control), but is slower end-to-end for large batches because you individually need to allocate infrastructure (spin up, down).
- Event-parallel (running a single subsampled key E2E) can deliver the fastest result for a single document but slowest throughput due to individual resource allocation.
- Pick based on operational needs; Metaxy handles metadata either way—the execution pattern is up to your orchestrator/resources.

## Best Practices

- Branch/subsample via fallbacks + `target_keys` to spin up fast, isolated experiments without touching prod.
- Root features: call `resolve_update(feature, samples=...)` with provenance; IOManager cannot supply samples.
- Downstream features: receive `Increment` via `ins=build_asset_in_from_feature(...)` and write metadata yourself.
- Always use `with store:` when reading/writing, and skip work when `Increment` is empty.

## Configuration

`MetaxyMetadataStoreResource.from_config` reads your existing `metaxy.toml`/`pyproject.toml`.
Pick a `store_name` and optional `fallback_stores` and you’re done (see the class docstring for full options).
No extra Dagster-specific config is required.
