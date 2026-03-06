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
    from metaxy.ext.metadata_stores.delta import DeltaMetadataStore

    store = DeltaMetadataStore(root_path="/path/to/directory")
    ```

2. Via Metaxy [configuration](../../reference/configuration.md):

    First, create a `metaxy.toml` file:

    ```toml title="metaxy.toml"
    [stores.dev]
    type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"
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

        config = mx.init()
        store = config.get_store("dev")
        ```


</div>

<!-- dprint-ignore-end -->

Now the `store` is ready to be used. We'll also assume there is a `MyFeature` [feature class](/guide/concepts/definitions/features.md) (1) prepared.
{ .annotate }

1. with `"my/feature"` key

## Writes

In order to save metadata into a metadata store, you can use the [`MetadataStore.write`][metaxy.MetadataStore.write] method:

!!! example

    ```py
    with store.open("w"):
        store.write(MyFeature, df)
    ```

Subsequent writes effectively overwrite the previous metadata, while actually [appending](../../metaxy/design.md#metadata-operations) to the same table.

--8<-- "flushing-metadata.md"

## Reads

Metadata can be retrieved using the [`MetadataStore.read`][metaxy.MetadataStore.read] method:

!!! example

    ```py
    with store.open("w"):
        df = store.write("my/feature", df)  # string keys work as well

    with store:
        df = store.read("my/feature")
    ```

By default, Metaxy drops historical records with the same feature version, which makes the `write`-`read` sequence idempotent for an outside observer.

## Increment Resolution

Increments can be computed using the [`MetadataStore.resolve_update`][metaxy.MetadataStore.resolve_update] method:

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

### Custom Staleness Conditions

By default, `resolve_update` only marks samples as stale when their upstream provenance has changed. The `staleness_predicates` parameter allows marking additional records as stale based on arbitrary conditions.
This is useful for forcing reprocessing after a bug fix that affected *metadata* (1), backfilling records that were processed with incomplete data, or invalidating samples that meet certain quality criteria.
{ .annotate }

1. and not *data*, because in this case you should be changing the feature version

Predicates are [Narwhals](https://narwhals-dev.github.io/narwhals/) expressions evaluated against stored metadata. When multiple predicates are provided, a sample is considered stale if it matches any of them.

!!! example "Reprocess failed or incomplete samples"

    <!-- skip next -->
    ```py
    import narwhals as nw

    with store.open("w"):
        inc = store.resolve_update(
            "my/feature",
            staleness_predicates=[
                nw.col("status") == "failed",
                nw.col("embedding_dim").is_null(),
            ],
        )
    ```

    Any record where `status` is `"failed"` **or** `embedding` is null will appear in `inc.stale`, even if its upstream provenance has not changed.

!!! tip "Where are increments computed?"

  Learn more [here](../../metaxy/design.md#compute).

!!! tip "How are increments computed?"

    Learn more [here](./versioning.md).

## Deletes

Metadata stores support deletions, which are not required during normal Metaxy operations (1).
{ .annotate }

1. deletions might be necessary when working with [expansion linear relationships](./definitions/relationship.md) and re-computing samples without changing the feature version

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

## Rebases

When a feature definition changes but the underlying computation stays the same (e.g., dependency graph refactoring, field renaming, code reorganization), existing metadata can be rebased onto the new feature version using [`MetadataStore.rebase`][metaxy.MetadataStore.rebase]. This recalculates provenance based on the target feature graph while preserving all user data columns.

`rebase` takes a dataframe of existing metadata, typically acquired with [`MetadataStore.read`][metaxy.MetadataStore.read] (1). The returned frame includes the target feature and project version columns, so pass `preserve_feature_version=True` to [`MetadataStore.write`][metaxy.MetadataStore.write] to retain them.
{ .annotate }

1. With non-default version filtering. See an example below.

!!! example

    <!-- skip next -->
    ```py
    from metaxy.models.constants import METAXY_FEATURE_VERSION

    with store.open("w"):
        existing = store.read(
            "example/child",
            with_feature_history=True,
            # an older feature version to be rebased
            filters=[nw.col(METAXY_FEATURE_VERSION) == "abc123"],
        )
        rebased = store.rebase(
            "example/child",
            existing,
            to_feature_version="def456",  # new feature version
        )
        store.write("example/child", rebased, preserve_feature_version=True)
    ```

Rebasing is also available as a CLI command with simplified options:

```console
$ metaxy metadata rebase example/child --from abc123 --to def456
```

!!! tip

    Use `--dry-run` to preview how many rows would be affected without writing the result.

## Fallback Stores

Metaxy metadata stores can be configured to pull missing metadata from another store. This is very useful for local and testing workflows, because it allows to avoid materializing the entire data pipeline locally.
Instead, Metaxy stores can automatically pull missing metadata from production.

Example Metaxy configuration:

```toml title="metaxy.toml"
[stores.dev]
type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"
root_path = "${HOME}/.metaxy/dev"
fallback_stores = ["prod"]

[stores.prod]
type = "metaxy.ext.metadata_stores.delta.DeltaMetadataStore"
root_path = "s3://my-prod-bucket/metaxy"
```

!!! warning

    Currently, the "missing metadata" detection works by checking whether the feature table exists in the store. This works in conjunction with [automatic table creation](), but doesn't work if empty tables are pre-created by e.g. migration tooling or some kind of CI/CD workflows. This will be improved in the future.

Metaxy doesn't mix metadata from different stores: either the entire feature is going to be pulled from the fallback store, or the primary store will be used.

Fallback stores can be chained at arbitrary depth.

## Metadata Store Implementations

Metaxy provides ready `MetadataStore` [implementations](../../integrations/metadata-stores/index.md) for popular databases and storage systems.
