# DuckDBMetadataStore

Metaxy implements [`DuckDBMetadataStore`][metaxy.metadata_store.duckdb.DuckDBMetadataStore]. It uses [DuckDB](https://duckdb.org/) as metadata storage and versioning engine.

## Installation

```shell
$ pip install 'metaxy[duckdb]'
```

## Extensions

DuckDB extensions can be loaded automatically:

```py
store = DuckDBMetadataStore("metadata.db", extensions=["hashfuncs", "spatial"])
```

`hashfuncs` is typically used by the versioning engine.

Learn more in the [API docs][metaxy.metadata_store.duckdb.DuckDBMetadataStore].
