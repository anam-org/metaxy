# Ibis Metadata Stores

The [`IbisMetadataStore`][metaxy.metadata_store.ibis.IbisMetadataStore] base class lets you plug any
[Edgeless SQL backend](https://docs.ibis-project.org/en/stable/backends.html) that supports structs. It powers
backend-specific integrations for ClickHouse, DuckDB, Postgres, and more.

## Configuration

::: metaxy-config
    class: metaxy.metadata_store.ibis.IbisMetadataStoreConfig
    path_prefix: stores.dev.config
    header_level: 3

## API Reference

- [Ibis API Reference](../../../../integrations/metadata-stores/databases/ibis/api.md)

## Backend-specific reference

- [Postgres backend reference](postgres.md)
