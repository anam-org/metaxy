---
title: "Metadata Store Integrations"
description: "Available metadata store backends for Metaxy."
---

# Metadata Stores

Metadata stores may come in two flavors.

## Database-Backed

These metadata stores provide **external compute** resources.
The most common example of such stores is databases.
Metaxy delegates all versioning computations and operations to external compute as much as possible. (1)
{ .annotate }

1. :fire: Typically (1) the entire [`MetadataStore.resolve_update`][metaxy.MetadataStore.resolve_update] can be executed externally!
   {.annotate}

   1. Except the cases enumerated in [../../guide/concepts/metadata-stores.md]

These metadata stores can be found [here](./databases/index.md).

!!! warning

    Metaxy does not handle infrastructure setup. Make sure to have large tables partitioned as appropriate for your use case.

!!! example

    [ClickHouse](./databases/clickhouse.md) is an excellent choice for a production metadata store.

!!! tip

    Some of them such as [LanceDB](./databases/lancedb.md) or [DuckDB](./databases/duckdb.md) can also act as local compute engines.

## Storage Only

These metadata stores only provide storage and rely on **local** (also referred to as **embedded**) **compute**.

The available storage-only stores can be found [here](./storage/index.md).

!!! example

    [DeltaLake](./storage/delta.md) is an excellent choice for a storage-only metadata store.

## Choosing the Right Metadata Store

Compute-backed stores are typically more performant, but require additional infrastructure and maintenance.

For **production** environments that need to handle **big metadata** volumes, consider database-backed stores.

For development, testing, branch deployments, and other scenarios where you want to keep things simple, consider using a storage-only store.

!!! warning

    Not all metadata stores support parallel writes.
    For example, using DuckDB with files requires [application level work-arounds](https://duckdb.org/docs/stable/connect/concurrency#writing-to-duckdb-from-multiple-processes).

## Reference

- Learn more about [using metadata stores](../../guide/concepts/metadata-stores.md)
