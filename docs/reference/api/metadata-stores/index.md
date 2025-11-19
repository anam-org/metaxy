# Metadata Stores

Metaxy abstracts interactions with metadata behind an interface called [MetadaStore][metaxy.metadata_store.base.MetadataStore].

Users can extend this class to implement support for arbitrary metadata storage such as databases, lakehouse formats, or really any kind of external system.
Metaxy has built-in support for the following metadata store types:

## Databases

See [IbisMetadataStore][metaxy.metadata_store.ibis.IbisMetadataStore].

## In-memory

See [InMemoryMetadataStore][metaxy.metadata_store.memory.InMemoryMetadataStore].

---

## Metadata Store Interface

::: metaxy.MetadataStore

::: metaxy.metadata_store.types.AccessMode
    members: true
    options:
      show_if_no_docstring: true

---

::: metaxy.metadata_store.base.VersioningEngineOptions
