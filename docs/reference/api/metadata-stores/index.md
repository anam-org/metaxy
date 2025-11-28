# Metadata Stores

Metaxy abstracts interactions with metadata behind an interface called [MetadaStore][metaxy.metadata_store.base.MetadataStore].

Users can extend this class to implement support for arbitrary metadata storage such as databases, lakehouse formats, or really any kind of external system.
Metaxy has built-in support for the following metadata store types:

## Databases

- [BigQuery](../../../integrations/metadata-stores/databases/bigquery/index.md)

- [ClickHouse](../../../integrations/metadata-stores/databases/clickhouse/index.md)

- [DuckDB](../../../integrations/metadata-stores/databases/duckdb/index.md)

- [LanceDB](../../../integrations/metadata-stores/databases/lancedb/index.md)

- [`IbisMetadataStore`][metaxy.metadata_store.ibis.IbisMetadataStore] (a base class) - see [Ibis integration](../../../integrations/metadata-stores/databases/ibis/index.md)

## Storage Only

- [DeltaMetadataStore](../../../integrations/metadata-stores/storage/delta/index.md)

- [InMemoryMetadataStore](./memory.md)

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

## Project Write Validation

By default, `MetadataStore` raises a `ValueError` when attempting to write to a project that doesn't match the expected project from `MetaxyConfig.get().project`.

For legitimate cross-project operations (such as migrations that need to update features across multiple projects), use `MetadataStore.allow_cross_project_writes`:

```python
with store.open("write"), store.allow_cross_project_writes():
    store.write_metadata(ExternallyDefinedFeature, df)
```
