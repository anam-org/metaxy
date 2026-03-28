---
title: "DuckDB Metadata Store"
description: "Learn how to use DuckDB as a Metaxy metadata store."
---

# Metaxy + DuckDB

[DuckDB](https://duckdb.org/) is an embedded analytical database. To use Metaxy with DuckDB, configure [`DuckDBMetadataStore`][metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore]. This runs versioning computations natively in DuckDB.

!!! warning

    File-based DuckDB does not (currently) support concurrent writes. If multiple writers are a requirement (e.g. with distributed data processing), consider using [Motherduck](https://motherduck.com/), [DuckLake](../storage/ducklake.md) with a `PostgreSQL` catalog, or refer to [DuckDB's documentation](https://duckdb.org/docs/stable/connect/concurrency#writing-to-duckdb-from-multiple-processes) to learn about implementing application-side work-arounds.

!!! tip "Recommended: enable [`Map` datatype](../../../guide/concepts/metadata-stores.md#map-datatype)"

    DuckDB natively supports the `Map` type. Enabling [`enable_map_datatype`](../../../reference/configuration.md#metaxy.config.MetaxyConfig.enable_map_datatype) preserves `Map` columns across read and write operations.

## Installation

```shell
pip install 'metaxy[duckdb]'
```

## API Reference

<!-- dprint-ignore-start -->
::: metaxy.ext.metadata_stores.duckdb
    options:
      members: false
      show_root_heading: true
      heading_level: 2

::: metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore
    options:
      members: false
      heading_level: 3

::: metaxy.ext.metadata_stores.duckdb.ExtensionSpec
    options:
      members: false
      heading_level: 3

<!-- dprint-ignore-end -->

## Configuration

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStoreConfig
    path_prefix: stores.dev.config
    header_level: 2
<!-- dprint-ignore-end -->
