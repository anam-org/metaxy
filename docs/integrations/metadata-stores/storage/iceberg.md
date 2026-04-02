---
title: "Metaxy + Apache Iceberg"
description: "Learn how to use Apache Iceberg to store Metaxy metadata."
---

# Metaxy + Apache Iceberg

[Apache Iceberg](https://iceberg.apache.org/) is an open table format for large analytic datasets supporting ACID transactions and schema evolution. Use [`IcebergMetadataStore`][metaxy.ext.polars.IcebergMetadataStore] to read and write Metaxy metadata from and to Iceberg tables. This metadata store is built on top of [PyIceberg](https://py.iceberg.apache.org/) and uses the in-memory Polars versioning engine for versioning computations.

!!! note
    By default, it uses a SQLite-backed SQL catalog for local development. You can configure any PyIceberg-supported catalog (REST, Glue, Hive) via [`catalog_properties`](#metaxy.ext.polars.handlers.iceberg.IcebergMetadataStoreConfig.catalog_properties).

!!! tip "Recommended: enable [`Map` datatype](../../../guide/concepts/metadata-stores.md#map-datatype)"

    Apache Iceberg supports the `Map` type natively. Enabling [`enable_map_datatype`](../../../reference/configuration.md#metaxy.config.MetaxyConfig.enable_map_datatype) preserves `Map` columns across read and write operations.

## Installation

```shell
pip install 'metaxy[iceberg]'
```

## API Reference

<!-- dprint-ignore-start -->
::: metaxy.ext.polars.handlers.iceberg
    options:
      members: false
      show_root_heading: true
      heading_level: 2

::: metaxy.ext.polars.IcebergMetadataStore
    options:
      members: false
      heading_level: 2
<!-- dprint-ignore-end -->

## Configuration

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.ext.polars.handlers.iceberg.IcebergMetadataStoreConfig
    path_prefix: stores.dev.config
    header_level: 3
<!-- dprint-ignore-end -->
