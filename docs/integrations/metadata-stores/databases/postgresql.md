---
title: "PostgreSQL Metadata Store"
description: "PostgreSQL as a metadata store backend."
---

# Metaxy + PostgreSQL

!!! warning "Experimental"
    This functionality is experimental.

Metadata managed by Metaxy can be stored in [`PostgreSQLMetadataStore`][metaxy.ext.postgresql.PostgreSQLMetadataStore].
It uses [PostgreSQL](https://www.postgresql.org/).
This metadata store backend is limited in comparison to others, because PostgreSQL doesn't support map-like data types, and Metaxy's versioning engine can't run in the database.
The local Polars versioning engine is used instead.
This results in the following limitations for [`MetadataStore.resolve_update`][metaxy.MetadataStore.resolve_update]:

- **Increased I/O**: entire upstream metadata has to be fetched to memory
- **Increased Memory footprint**: expect high memory usage, especially when having many upstream features

## Metaxy's Versioning Columns

PostgreSQL doesn't have native map-like or struct types, so Metaxy's versioning columns are stored as `JSONB`.
Metaxy's [`Map`](../../../guide/concepts/metadata-stores.md#map-datatype) versioning columns are decomposed into named `Struct` columns, JSON-encoded for storage, and reconstructed as `Map` on read, so callers always see `Map` columns.
As a convenience feature, `PostgreSQLMetadataStore` also automatically json-encodes user-defined `pl.Struct` columns when writing metadata and parses them back to `pl.Struct` when reading.
This convenience for user columns can be disabled with the [`auto_cast_struct_for_jsonb`](#metaxy.ext.postgresql.PostgreSQLMetadataStoreConfig.auto_cast_struct_for_jsonb) configuration parameter; Metaxy's versioning columns are always converted regardless.

## API Reference

::: metaxy.ext.postgresql
    options:
      members: false
      show_root_heading: true
      heading_level: 2

::: metaxy.ext.postgresql.PostgreSQLMetadataStore
    options:
      inherited_members: false
      heading_level: 3

## Configuration

::: metaxy-config
    class: metaxy.ext.postgresql.PostgreSQLMetadataStoreConfig
    path_prefix: stores.dev.config
    header_level: 3
