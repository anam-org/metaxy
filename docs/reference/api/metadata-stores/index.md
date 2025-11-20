# Metadata Stores

Metaxy abstracts interactions with metadata behind an interface called [MetadaStore][metaxy.metadata_store.base.MetadataStore].

Users can extend this class to implement support for arbitrary metadata storage such as databases, lakehouse formats, or really any kind of external system.
Metaxy has built-in support for the following metadata store types:

## Databases

[IbisMetadataStore][metaxy.metadata_store.ibis.IbisMetadataStore].

## Storage Only

- [DeltaMetadataStore](./delta.md).
- [InMemoryMetadataStore](./memory.md).

---

## Metadata Store Interface

::: metaxy.MetadataStore

::: metaxy.metadata_store.types.AccessMode
    members: true
    options:
      show_if_no_docstring: true

---

::: metaxy.metadata_store.base.VersioningEngineOptions
