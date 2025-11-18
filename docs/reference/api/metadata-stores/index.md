# Metadata Stores

Metaxy abstracts interactions with metadata behind an interface called [MetadaStore][metaxy.metadata_store.base.MetadataStore].

Users can extend this class to implement support for arbitrary metadata storage such as databases, lakehouse formats, or really any kind of external system.
Metaxy has built-in support for the following metadata store types:

## Databases

- [IbisMetadataStore][metaxy.metadata_store.ibis.IbisMetadataStore].
- [LanceDBMetadataStore](./lancedb.md).

## Storage Only

- [DeltaMetadataStore](./delta.md).
- [InMemoryMetadataStore](./memory.md).

---

## Metadata Store Interface

::: metaxy.MetadataStore

::: metaxy.metadata_store.types.AccessMode
    options:
      show_if_no_docstring: true

---

::: metaxy.metadata_store.base.VersioningEngineOptions


---

## Project Write Validation

By default, `MetadataStore` raises a `ValueError` when attempting to write to a project that doesn't match the expected project from `MetaxyConfig.get().project`.

For legitimate cross-project operations (such as migrations that need to update features across multiple projects), use `MetadataStore.allow_cross_project_writes`:

```python
with store.open("write"), store.allow_cross_project_writes():
    store.write_metadata(ExternallyDefinedFeature, df)
```
