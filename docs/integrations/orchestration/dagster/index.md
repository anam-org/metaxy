---
title: "Dagster Integration"
description: "Dagster integration for Metaxy."
---

# Metaxy + Dagster

Metaxy's dependency system has been originally inspired by [Dagster](https://dagster.io/).

Because of this, Metaxy code can be naturally composed with Dagster code, Metaxy concepts map directly into Dagster concepts, and the provided [`@metaxify`][metaxy.ext.dagster.metaxify.metaxify] decorator makes this process effortless.

The only step that has to be taken in order to inject Metaxy into Dagster assets is to associate the Dagster asset with the Metaxy feature.

Unleash the full power of `@metaxify` on Dagster!

<!-- dprint-ignore-start -->
!!! example

    Set the well-known `"metaxy/feature"` key (1):
    { .annotate }

    1. :man_raising_hand: point it to... the Metaxy feature key!

    ```python
    import dagster as dg
    import metaxy.ext.dagster as mxd

    @mxd.metaxify()
    @dg.asset(metadata={"metaxy/feature": "my/metaxy/feature"})
    def my_asset():
      ...
    ```
<!-- dprint-ignore-end -->

`@metaxify` will take care of injecting information (such as asset dependencies or metadata) from the Metaxy feature to the Dagster asset.
Learn more about `@metaxify` (with example screenshots) [here](metaxify.md).

## What's in the box

This integration provides:

- [`metaxify`][metaxy.ext.dagster.metaxify.metaxify] - a decorator that enriches Dagster asset definitions with Metaxy information such as upstream dependencies, description, metadata, code version, table schema, column lineage, and so on. More info and some screenshots [here](metaxify.md).

- [`MetaxyIOManager`][metaxy.ext.dagster.io_manager.MetaxyIOManager] - an IO manager that reads and writes Dagster assets that are Metaxy features and logs useful runtime metadata.

- [`MetaxyStoreFromConfigResource`][metaxy.ext.dagster.MetaxyStoreFromConfigResource] - a resource that provides access to [`MetadataStore`][metaxy.MetadataStore]

- [`generate_materialize_results`][metaxy.ext.dagster.utils.generate_materialize_results] / [`generate_observe_results`][metaxy.ext.dagster.utils.generate_observe_results] - generators for yielding `dagster.MaterializeResult` or `dagster.ObserveResult` events from Dagster assets (and multi-assets), with automatic topological ordering, partition filtering, logging row counts, and setting [Dagster data versions](https://docs.dagster.io/guides/build/assets/asset-versioning-and-caching#step-three-computing-your-own-data-versions).

- [`observable_metaxy_asset`][metaxy.ext.dagster.observable.observable_metaxy_asset] - a decorator that creates observable source assets for monitoring external Metaxy features.

## Quick Start

### 1. Define Metaxy Features

<!-- dprint-ignore-start -->
```python {title="defs.py"}
--8<-- "example-dagster/src/example_dagster/definitions.py:feature-definitions"
```
<!-- dprint-ignore-end -->

### 2. Define Dagster Assets

<!-- dprint-ignore-start -->
!!! example "Root Asset"

    Let's define an asset that doesn't have any upstream Metaxy features.

    ```python {title="defs.py"}
    --8<-- "example-dagster/src/example_dagster/definitions.py:root-asset"
    ```
<!-- dprint-ignore-end -->

<!-- dprint-ignore-start -->
!!! example "Downstream Asset"

    ```py {title="defs.py"}
    --8<-- "example-dagster/src/example_dagster/definitions.py:downstream-asset"
    ```
<!-- dprint-ignore-end -->

<!-- dprint-ignore-start -->
!!! example "Non-Metaxy Downstream Asset"

    ```py
    --8<-- "example-dagster/src/example_dagster/definitions.py:non-metaxy-downstream"
    ```
<!-- dprint-ignore-end -->

### 3. Create Dagster Definitions

<!-- dprint-ignore-start -->
```py {title="defs.py"}
--8<--  "example-dagster/src/example_dagster/definitions.py:dagster-definitions"
```
<!-- dprint-ignore-end -->

1. This loads Metaxy configuration and feature definitions

### 4. Start Dagster

```bash
dg dev -f defs.py
```

Materialize your assets and let Metaxy take care of state and versioning!

## Observable Source Assets

Use [`observable_metaxy_asset`][metaxy.ext.dagster.observable.observable_metaxy_asset] to create observable source assets that monitor external Metaxy features.
This is useful when Metaxy features are populated outside of Dagster (e.g., by external pipelines) and you want Dagster to track their data versions.

<!-- dprint-ignore-start -->
!!! example "Basic Observable Asset"

    ```python
    import dagster as dg
    import metaxy as mx
    import metaxy.ext.dagster as mxd

    @mxd.observable_metaxy_asset(key="dagster/asset/key", feature="external/feature")
    def external_data(context, store: dg.ResourceParam[mx.MetadataStore], lazy_df: nw.LazyFrame):
        # build a custom metadata dict
        metadata = ...
        return metadata
    ```
<!-- dprint-ignore-end -->

The observation automatically tracks:

- **Data version**: Uses `mean(metaxy_created_at)` to detect both additions and deletions
- **Row count**: Logged as `dagster/row_count` metadata

## Deletion workflows with Dagster ops

The Dagster integration provides a `delete_metadata` op:

```python
import dagster as dg
import metaxy.ext.dagster as mxd


# Define a job with the delete op
@dg.job(resource_defs={"metaxy_store": mxd.MetaxyStoreFromConfigResource(name="default")})
def cleanup_job():
    mxd.delete_metadata()


# Execute with config to delete inactive customer segments
cleanup_job.execute_in_process(
    run_config={
        "ops": {
            "delete_metadata": {
                "config": {
                    "feature_key": ["customer", "segment"],
                    "filters": ["status = 'inactive'"],
                    "soft": True,
                }
            }
        }
    }
)
```

`filters` is a list of SQL WHERE clause strings (e.g., `["status = 'inactive'", "age > 18"]`) that are parsed into Narwhals expressions. Multiple filters are combined with AND logic. See the [filter expressions guide](../../../guide/learn/filters.md) for supported syntax.
Set `soft=False` to physically remove rows.

## Reference

- [API](api.md)
- Dagster [docs](https://docs.dagster.io/)
