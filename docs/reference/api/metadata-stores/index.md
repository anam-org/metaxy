---
title: "Metadata Stores API"
description: "API reference for the MetadataStore interface."
---

# Metadata Stores

Metaxy abstracts interactions with metadata behind an interface called [`MetadataStore`][metaxy.metadata_store.base.MetadataStore].

Users can extend this class to implement support for arbitrary metadata storage such as databases, lakehouse formats, or really any kind of external system.

Learn how to use metadata stores [here](../../../guide/learn/metadata-stores.md).

Here are some of the built-in metadata store types (1):
{ .annotate }

1. the full list can be found [here](../../../integrations/metadata-stores/index.md)

## Databases

- [BigQuery](../../../integrations/metadata-stores/databases/bigquery.md)

- [ClickHouse](../../../integrations/metadata-stores/databases/clickhouse.md)

- [DuckDB](../../../integrations/metadata-stores/databases/duckdb.md) - if used with Motherduck

- [LanceDB](../../../integrations/metadata-stores/databases/lancedb.md) - if used with LanceDB Cloud

## Storage

- [DeltaMetadataStore](../../../integrations/metadata-stores/storage/delta.md)

- [DuckDB](../../../integrations/metadata-stores/databases/duckdb.md) - if used with a local file

- [LanceDB](../../../integrations/metadata-stores/databases/lancedb.md) - if used with a local file

---

## Metadata Store Interface

::: metaxy.MetadataStore

::: metaxy.metadata_store.types.AccessMode
    options:
      show_if_no_docstring: true

::: metaxy.metadata_store.base.VersioningEngineOptions

---

## Base Configuration Class

The following base configuration class is typically used by child metadata stores:

::: metaxy.metadata_store.base.MetadataStoreConfig

---

## Configuration

The base [`MetadataStoreConfig`][metaxy.metadata_store.base.MetadataStoreConfig] class injects the following configuration options:

::: metaxy-config
    class: metaxy.metadata_store.base.MetadataStoreConfig
    path_prefix: stores.dev.config
    header_level: 3
