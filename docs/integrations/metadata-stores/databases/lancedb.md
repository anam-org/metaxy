---
title: "LanceDB Metadata Store"
description: "LanceDB as a metadata store backend."
---

# LanceDB

!!! warning "Experimental"

    This functionality is experimental.

[LanceDB](https://lancedb.github.io/lancedb/) is an vector database built on the Lance columnar format. To use Metaxy with LanceDB, configure [`LanceDBMetadataStore`][metaxy.ext.metadata_stores.lancedb.LanceDBMetadataStore]. It uses the in-memory Polars engine for versioning computations. LanceDB handles schema evolution, transactions, and compaction automatically.

It runs embedded (local directory) or against external storage (object stores, HTTP endpoints, LanceDB Cloud), so you can use the same store type for local development and cloud workloads.

## Installation

The backend relies on [`lancedb`](https://lancedb.com/), which is shipped with Metaxy's `lancedb` extras.

```shell
pip install 'metaxy[lancedb]'
```

## Storage Targets

Point `uri` at any supported URI (`s3://`, `gs://`, `az://`, `db://`, ...) and forward credentials with the platform's native mechanism (environment variables, IAM roles, workload identity, etc.). LanceDB supports local filesystem, S3, GCS, Azure, LanceDB Cloud, and remote HTTP/HTTPS endpoints.

## Storage Layout

All tables are stored within a single LanceDB database at the configured URI location.
Each feature gets its own Lance table.

---

<!-- dprint-ignore-start -->
::: metaxy.ext.metadata_stores.lancedb
    options:
      members: false
      show_root_heading: true
      heading_level: 2

::: metaxy.ext.metadata_stores.lancedb.LanceDBMetadataStore
    options:
      members: false
      heading_level: 3
<!-- dprint-ignore-end -->

## Configuration

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.ext.metadata_stores.lancedb.LanceDBMetadataStoreConfig
    path_prefix: stores.dev.config
    header_level: 3
<!-- dprint-ignore-end -->
