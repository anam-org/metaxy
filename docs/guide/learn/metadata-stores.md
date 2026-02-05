---
title: "Metadata Stores"
description: "Learn how to use Metadata Stores."
---

# Metadata Stores

Metaxy abstracts interactions with metadata stored in external systems such as databases, files, or object stores, through a unified interface: [`MetadataStore`][metaxy.MetadataStore]. `MetadataStore` is implemented to satisfy [storage design choices](../../metaxy/design.md#storage).

All operations with metadata stores may reference features as one of the supported [syntactic sugar](./syntactic-sugar.md#features) alternatives. In practice, it is typically convenient to either use feature classes or stringified feature keys.

Metadata accept [Narwhals-compatible](https://narwhals-dev.github.io/narwhals/) dataframes and return Narwhals dataframes. In practice, we have tested Metaxy with Pandas, Polars and Ibis dataframes.

## Instantiation

There are generally two ways to create a `MetadataStore`. We are going to demonstrate both with [DeltaLake](../../integrations/metadata-stores/storage/delta.md) as an example.

<!-- dprint-ignore-start -->

<div class="annotate" markdown>

1. Using the Python API directly:

    ```py
    from metaxy.metadata_store.delta import DeltaMetadataStore

    store = DeltaMetadataStore(root_path="/path/to/directory")
    ```

2. Via Metaxy [configuration](../../reference/configuration.md):

    First, create a `metaxy.toml` file (1):

    ```toml title="metaxy.toml"
    [stores.dev]
    type = "metaxy.metadata_store.delta.DeltaMetadataStore"
    root_path = "/path/to/directory"
    ```

    Now the metadata store can be constructed from a [`MetaxyConfig`][metaxy.MetaxyConfig] instance.

    === "Metaxy Already Initialized"

        ```py
        import metaxy as mx

        config = mx.MetaxyConfig.get()
        store = config.get_store("dev")
        ```

    === "With Metaxy Initialization"
        <!-- skip next -->
        ```py
        import metaxy as mx

        config = mx.init_metaxy()
        store = config.get_store("dev")
        ```


</div>

<!-- dprint-ignore-end -->

1. `[tool.metaxy]` section in `pyproject.toml` is supported as well

Now the `store` is ready to be used. We'll also assume there is a `MyFeature` [feature class](/guide/learn/definitions/features.md) (1) prepared.
{. annotate }

1. with `"my/feature"` key

## Writes

In order to save metadata into a metadata store, you can use the [`write`][metaxy.MetadataStore.write] method:

!!! example

    ```py
    with store.open("w"):
        store.write(MyFeature, df)
    ```

Subsequent writes effectively overwrite the previous metadata, while actually [appending](../../metaxy/design.md#metadata-operations) to the same table.

## Reads

Metadata can be retrieved using the [`read`][metaxy.MetadataStore.read] method:

!!! example

    ```py
    with store.open("w"):
        df = store.write("my/feature", df)  # string keys work as well

    with store:
        df = store.read("my/feature")
    ```

## Increment Resolution

Increments can be computed using the [`resolve_update`][metaxy.MetadataStore.resolve_update] method:

!!! example

    <!-- skip next -->
    ```py
    with store.open("w"):
        inc = store.resolve_update("my/feature")
    ```

The returned [`Increment`][metaxy.Increment] (or [`LazyIncrement`][metaxy.LazyIncrement]) holds fresh samples that haven't been processed yet, stale samples which require to be processed again, and orphaned samples which are no longer present in upstream features and may be deleted.

!!! tip annotate

    Root features (1) require the `samples` argument to be set as well, since Metaxy would not be able to load upstream metadata automatically.

1. features that do not have upstream features

It is up to the caller to decide how to handle the processing and potential deletion of orphaned samples.

Once processing is complete, the caller is expected to call `MetadataStore.write` to record metadata about the processed samples.

!!! tip

    Learn more about how increments are computed [here](../../metaxy/design.md#compute).

## Deletes

Metadata stores support deletions, which are not required during normal Metaxy operations (1).

1. deletions might be necessary when working with [expansion linear relationships](./definitions/relationship.md).

Here is an example of how a deletion would look like:

```py
from datetime import datetime, timedelta, timezone

import narwhals as nw

with store.open("w"):
    store.delete(
        MyFeature,
        filters=[nw.col("metaxy_created_at") < datetime.now(timezone.utc) - timedelta(days=30)],
    )
```

Learn more about deletions [here](./deletions.md).

## Fallback Stores

Metaxy metadata stores can be configured to pull missing metadata from another store. This is very useful for local and testing workflows, because it allows to avoid materializing the entire data pipeline locally.
Instead, Metaxy stores can automatically pull missing metadata from production.

Example Metaxy configuration:

```toml title="metaxy.toml"
[stores.dev]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"
root_path = "${HOME}/.metaxy/dev"
fallback_stores = ["prod"]

[stores.prod]
type = "metaxy.metadata_store.delta.DeltaMetadataStore"
root_path = "s3://my-prod-bucket/metaxy"
```

!!! warning

    Currently, the "missing metadata" detection works by checking whether the feature table exists in the store. This works in conjunction with [automatic table creation](), but doesn't work if empty tables are pre-created by e.g. migration tooling or some kind of CI/CD workflows. This will be improved in the future.

Metaxy doesn't mix metadata from different stores: either the entire feature is going to be pulled from the fallback store, or the primary store will be used.

Fallback stores can be chained at arbitrary depth.

## Metadata Store Implementations

Metaxy provides ready `MetadataStore` [implementations](../../integrations/metadata-stores/index.md) for popular databases and storage systems.
