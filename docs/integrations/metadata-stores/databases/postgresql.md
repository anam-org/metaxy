---
title: "PostgreSQL Metadata Store"
description: "PostgreSQL as a metadata store backend."
---

# Metaxy + PostgreSQL

!!! warning "Experimental"
    This functionality is experimental.

Metadata managed by Metaxy can be stored in [`PostgreSQLMetadataStore`][metaxy.metadata_store.postgresql.PostgreSQLMetadataStore].
It uses [PostgreSQL](https://www.postgresql.org/).
This metadata store backend is limited in comparison to others, because PostgreSQL doesn't support map-like data types, and Metaxy's versioning engine can't run in the database.
The local Polars versioning engine is used instead.
This results in the following limitations for [`MetadataStore.resolve_update`][metaxy.MetadataStore.resolve_update]:

- **Increased I/O**: entire upstream metadata has to be fetched to memory
- **Increased Memory footprint**: expect high memory usage, especially when having many upstream features

## Metaxy's Versioning Struct Columns

PostgreSQL doesn't have native map-like or struct types, so it's recommended to store Metaxy's versioning columns as `JSONB`.
As a convenience feature, `PostgreSQLMetadataStore` will automatically json-encodes `pl.Struct` columns when writing metadata and parse them to `pl.Struct` when reading.
This behavior can be disabled with `auto_cast_struct_for_jsonb` configuration parameter. This setting only affects user-defined columns, while Metaxy's versioning columns are always encoded/parsed.

## API Reference

::: metaxy.metadata_store.postgresql
    options:
      members: false
      show_root_heading: true
      heading_level: 2

::: metaxy.metadata_store.postgresql.PostgreSQLMetadataStore
    options:
      inherited_members: false
      heading_level: 3

## Configuration

::: metaxy-config
    class: metaxy.metadata_store.postgresql.PostgreSQLMetadataStoreConfig
    path_prefix: stores.dev.config
    header_level: 3
