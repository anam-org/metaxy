---
title: "Design Choices"
description: "Explanation of design choices and architectural decisions in Metaxy."
---

# Design Choices

As discussed on the [front page](../index.md) and [the Pitch](./pitch.md), Metaxy aims to be pluggable, reliable, scalable, and developer-friendly. Here are some of the design choices we made to achieve these goals.

## Storage

--8<-- "data-vs-metadata.md"

Metaxy is designed to be compatible with storage systems which satisfy the following requirements:

<div class="annotate" markdown>

- has an _append_ operation

- can store _map-like elements_ (e.g. dictionaries)

    ??? note "Lifting This Requirement"

        Unfortunately, the most popular database - PostgreSQL - does not satisfy it.
        While PostgreSQL is not an ideal choice for a Metaxy [Metadata Store](../guide/concepts/metadata-stores.md)
        for other reasons (mainly being analytical queries performance), we recognize the need to support it and are exploring a solution in [`anam-org/metaxy#223`](https://github.com/anam-org/metaxy/issues/223).
        This requirement may not be necessary in the future.

</div>


This allows Metaxy to target modern data warehouses (e.g. ClickHouse, BigQuery, Snowflake, or the more minimalistic DuckDB) and storage formats such as DeltaLake, Iceberg, DuckLake, and anything compatible with Apache Arrow.

The Metaxy abstraction that implements these design choices and is used to interact with storage systems is known as [Metadata Store](../guide/concepts/metadata-stores.md).

### Table Schema

Metaxy uses the same storage layout for all storage systems. Each [feature](../guide/concepts/definitions/features.md) gets a separate table.

Here is how a typical Metaxy feature table looks like:

--8<-- "table.md"

!!! info

    `metaxy_data_version`/`metaxy_data_version_by_field` and `metaxy_provenance`/`metaxy_provenance_by_field` serve a slightly different purpose.
    Provenance columns hold **static** versioning information entirely defined by the Metaxy framework. Data version defaults to the same value as provenance, but can be customized by the user **at runtime**, for example derived from the **contents** of the computed sample. Learn more [here](../guide/concepts/data-versioning.md).

All historical records for a given feature are stored in the same table. They can be separated by the following [system columns](../guide/concepts/system-columns.md):

- `metaxy_feature_version` is shared among multiple rows and is changed on any of the feature or upstream feature `code_version` changes

- `metaxy_data_version`, `metaxy_data_version_by_field`, `metaxy_provenance`, `metaxy_provenance_by_field` carry versioning and provenance information about the specific row

- `metaxy_created_at`, `metaxy_updated_at`, `metaxy_deleted_at` allow to identify the latest active row for a given feature version

### Metadata Operations

Metaxy tables are **immutable**. Once written, a row is never modified or deleted (1).
{ .annotate }

1. but users can [delete rows](../guide/concepts/deletions.md#hard-deletes) manually if needed

As discussed earlier, writing metadata in Metaxy is done by **appending** to a feature table. Subsequent writes with the same feature version effectively act as overwrites. This is achieved by filtering out older rows using the `metaxy_updated_at` columns (1). [Soft-deletes](../guide/concepts/deletions.md#soft-deletes) are implemented as appends as well.
{ .annotate }

1. also known as merge-on-read

The append-only design choice has a few significant benefits:

- unlocks easier and lock-free setups for multiple writers

- ensures existing and historical metadata can never be lost or corrupted

!!! tip

    Users can implement [storage cleanup](../guide/concepts/deletions.md#hard-deletes) based on their specific needs and constraints.

- avoids additional write-time checks or operations, which has performance benefits

- allows Metaxy to be used with storage systems which lack ACID guarantees and do not support transactions

---

!!! info

    These design choices come with a cost of increased storage usage. But storage is cheap while mistakes aren't.

## DataFrame API

In order to be versatile and support different compute engines, Metaxy uses [Narwhals](https://github.com/narwhals/narwhals) for DataFrame manipulations.

This allows metadata store implementations to reuse the same code. Currently only a [thin subset][metaxy.metadata_store.MetadataStore] requires storage-specific implementations (1).
{ .annotate }

(1) such as providing database-specific hashing syntax and some other operations

## Compute

Increment resolution in Metaxy involves running computations: every time the user requests an increment for a given [feature](../guide/concepts/definitions/features.md), Metaxy has to join upstream features, hash their versions, and filter out samples that have already been processed. This can be performed either **locally** (typically favored in development environments) or **remotely** (achieves better performance in production). Metaxy supports both options: [databases](../integrations/metadata-stores/databases/index.md) for remote compute and [storage-only](../integrations/metadata-stores/storage/index.md) metadata stores for embedded compute (1).
{ .annotate }

1. e.g. Polars or Duckdb

When resolving incremental updates for a [feature](../guide/concepts/definitions/features.md), Metaxy attempts to perform all computations such as [sample version calculations](../guide/concepts/data-versioning.md) within the metadata store.

!!! note "When can **local** computations happen instead"

    Metaxy's versioning engine runs on the **local Polars versioning engine** if:

    1. The metadata store does not have a compute engine at all: for example, [DeltaLake](https://delta.io/) is just a storage format.

    2. The user explicitly requested to keep the computations **local** by setting `versioning_engine="polars"` when instantiating the metadata store.

    3. A **fallback store** had to be used to retrieve one of the parent features missing in the current store.

    All 3 cases cannot be accidental and require preconfigured settings or explicit user action. In the third case, Metaxy will also issue a warning just in case the user has accidentally configured a fallback store in production.

All metadata store implementations are guaranteed to return equivalent results. They are continuously tested against the reference Polars implementation.

## 🚀 What's Next?

- Itching to write some Metaxy code? Jump to [Quickstart](/guide/quickstart/quickstart.md).
--8<-- "whats-next.md"
