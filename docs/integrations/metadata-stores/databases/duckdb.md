---
title: "DuckDB Metadata Store"
description: "DuckDB as a metadata store backend."
---

# DuckDB

[DuckDB](https://duckdb.org/) is an embedded analytical database. To use Metaxy with DuckDB, configure [`DuckDBMetadataStore`][metaxy.metadata_store.duckdb.DuckDBMetadataStore]. This runs versioning computations natively in DuckDB.

!!! warning

    File-based DuckDB does not (currently) support concurrent writes. If multiple writers are a requirement (e.g. with distributed data processing), consider either using DuckLake with a `PostgreSQL` catalog, or refer to [DuckDB's documentation](https://duckdb.org/docs/stable/connect/concurrency#writing-to-duckdb-from-multiple-processes) to learn about implementing application-side work-arounds.

!!! tip

    The [Delta Lake metadata store](../storage/delta.md) might be a better alternative for concurrent writes (with it's Polars-based versioning engine being as fast as DuckDB).

## Installation

```shell
pip install 'metaxy[duckdb]'
```

## Extensions

DuckDB extensions can be loaded automatically:

```py
store = DuckDBMetadataStore("metadata.db", extensions=["hashfuncs", "spatial"])
```

`hashfuncs` is typically used by the versioning engine.

---

<!-- dprint-ignore-start -->
::: metaxy.metadata_store.duckdb
    options:
      members: false
      show_root_heading: true
      heading_level: 2

::: metaxy.metadata_store.duckdb.DuckDBMetadataStore
    options:
      members: false
      heading_level: 3

::: metaxy.metadata_store.duckdb.ExtensionSpec
    options:
      members: false
      heading_level: 3

::: metaxy.metadata_store.duckdb.DuckLakeConfigInput
    options:
      members: false
      heading_level: 3

::: metaxy.metadata_store._ducklake_support.DuckLakeAttachmentConfig
    options:
      members: false
      heading_level: 3
<!-- dprint-ignore-end -->

## Configuration

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.metadata_store.duckdb.DuckDBMetadataStoreConfig
    path_prefix: stores.dev.config
    header_level: 2
<!-- dprint-ignore-end -->
