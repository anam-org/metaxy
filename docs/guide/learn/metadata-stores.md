---
title: "Metadata Stores"
description: "How Metaxy abstracts metadata storage and versioning."
---

# Metadata Stores

Metaxy abstracts interactions with metadata stored in external systems such as databases, files, or object stores, through a unified interface: [`MetadataStore`][metaxy.MetadataStore].

Metadata stores expose methods for [reading][metaxy.MetadataStore.read_metadata], [writing][metaxy.MetadataStore.write_metadata], deleting metadata, and the most important one: [resolve_update][metaxy.MetadataStore.resolve_update] for receiving a metadata increment.

It looks more or less like this:

!!! example

    ```py
    with store:
        df = store.read_metadata("/my/feature/key")

    with store.open("write"):
        store.write_metadata("/another/key", df)

    with store:
        increment = store.resolve_update("and/another/key")
    ```

Metadata stores implement an append-only storage model and rely on [Metaxy system columns](system-columns.md).

!!! note

    Metaxy never mutates metadata in-place (1)
    { .annotate }

    1. :fire: safety and performance reasons

!!! warning "Forged About ACID"

    Metadata reads/writes **are not guaranteed to be ACID**: Metaxy is designed to interact with analytical databases which lack ACID guarantees by definition and design. (1)
    { .annotate }

    1. for - you've guessed it right - :fire: performance reasons

    However, Metaxy never retrieves the same sample version twice, and performs read-time deduplication (1) by the combination of the feature version, ID columns, and `metaxy_created_at`.
    { .annotate }

    1. also known as **merge-on-read**

When resolving incremental updates for a [feature](feature-definitions.md), Metaxy attempts to perform all computations such as [sample version calculations](data-versioning.md) within the metadata store.
This includes joining upstream features, hashing their versions, and filtering out samples that have already been processed - everything is pushed into the DB.

!!! note "When can **local** computations happen instead"

    Metaxy's versioning engine runs **locally** instead:

    !!! info

        The **local** versioning engine is implemented with [`polars-hash`](https://github.com/ion-elgreco/polars-hash) and benefits from parallelism, predicate pushdown, and other features of [Polars](https://pola.rs/).

    1. If the metadata store does not have a compute engine at all: for example, [DeltaLake](https://delta.io/) is just a storage format.

    2. If the user explicitly requested to keep the computations **local** by setting `versioning_engine="polars"` when instantiating the metadata store.

    3. If a **fallback store** had to be used to retrieve one of the parent features missing in the current store.

    All 3 cases cannot be accidental and require preconfigured settings or explicit user action. In the third case, Metaxy will also issue a warning just in case the user has accidentally configured a fallback store in production.

## Deletions

Deletes are typically not required during normal operations, but they are still supported for cleanup purposes. (1)
{ .annotate }

1. deletions might be necessary when working with [expansion linear relationships](relationship.md).

Here is an example of how a deletion would look like:

```py
from datetime import datetime, timedelta

import narwhals as nw

with store.open("write"):
    store.delete_metadata(
        "my/feature",
        filters=[nw.col("metaxy_created_at") < datetime.now() - timedelta(days=30)],
    )
```

## Metadata Store Implementations

Metaxy provides ready `MetadataStore` [implementations](../../integrations/metadata-stores/index.md) for popular databases and storage systems.
