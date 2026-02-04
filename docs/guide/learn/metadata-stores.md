---
title: "Metadata Stores"
description: "Learn how to use Metadata Stores."
---

# Metadata Stores

Metaxy abstracts interactions with metadata stored in external systems such as databases, files, or object stores, through a unified interface: [`MetadataStore`][metaxy.MetadataStore]. `MetadataStore` is implemented to satisfy [storage design choices](../../metaxy/design.md#storage).

All operations with metadata stores may reference features as one of the supported [syntactic sugar](./syntactic-sugar.md#features) alternatives. In practice, it is typically convenient to either use feature classes or stringified feature keys.

Metadata accept [Narwhals-compatible](https://narwhals-dev.github.io/narwhals/) dataframes and return Narwhals dataframes.

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

## Resolving Increments

Increments can be computed using the [`resolve_update`][metaxy.MetadataStore.resolve_update] method:

!!! example

    <!-- skip next -->
    ```py
    with store.open("w"):
        inc = store.resolve_update("my/feature")
    ```

The returned [`Increment`][metaxy.Increment] or [`LazyIncrement`][metaxy.LazyIncrement] contains fresh samples that haven't been processed yet, stale samples which require to be processed again, and orphaned samples which are no longer present in upstream features and may be deleted.

It is up to the caller to decide how to handle the processing and potential deletion of orphaned samples.

Once processing is complete, the caller is expected to call `MetadataStore.write` to record metadata about the processed samples.

!!! tip

    Learn more about how increments are computed [here](../../metaxy/design.md#compute).

## Deletions

Metadata stores support deletions, which are not required during normal Metaxy operations (1).

1. deletions might be necessary when working with [expansion linear relationships](relationship.md).

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

## Metadata Store Implementations

Metaxy provides ready `MetadataStore` [implementations](../../integrations/metadata-stores/index.md) for popular databases and storage systems.
