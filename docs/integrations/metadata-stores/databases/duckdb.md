# Metaxy + DuckDB

Metaxy implements [`DuckDBMetadataStore`][metaxy.metadata_store.duckdb.DuckDBMetadataStore]. It uses [DuckDB](https://duckdb.org/) as metadata storage and versioning engine.

!!! warning

    DuckDB does not (currently) support concurrent writes. If multiple writers are a requirement (e.g. with distributed data processing), consider either using DuckLake with a `PostgreSQL` catalog, or refer to [DuckDB's documentation](https://duckdb.org/docs/stable/connect/concurrency#writing-to-duckdb-from-multiple-processes) to learn about implementing application-side work-arounds.

!!! tip

    The [Delta Lake metadata store](../storage/delta.md) might be a better alternative for concurrent writes.

## Installation

```shell
pip install 'metaxy[duckdb]'
```

## Extensions

DuckDB extensions can be loaded automatically:

```py
store = DuckDBMetadataStore("metadata.db", extensions=["hashfuncs", "spatial"])
```

`hashfuncs` is typically used by the versioning engine.

# Reference

- [API docs][metaxy.metadata_store.duckdb.DuckDBMetadataStore].
