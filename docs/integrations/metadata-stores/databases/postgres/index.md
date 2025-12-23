# Metaxy + PostgreSQL

Metaxy implements [`PostgresMetadataStore`][metaxy.metadata_store.postgres.PostgresMetadataStore].
PostgreSQL provides metadata storage with JSONB columns for struct-like data, enabling full SQL query capabilities.

The store uses Ibis for lazy query execution and supports both MD5 (built-in) and SHA256 (via pgcrypto extension) hash algorithms.

## Installation

The backend relies on [`psycopg`](https://www.psycopg.org/), which is shipped with Metaxy's `postgres` extras.

```shell
pip install 'metaxy[postgres]'
```

## Storage Layout

Each feature gets its own PostgreSQL table within the configured database and schema. Struct columns (`metaxy_provenance_by_field`, `metaxy_data_version_by_field`) are stored as JSONB columns and automatically packed/unpacked during read/write operations.

## Reference

- [Configuration](configuration.md)
- [API](api.md)
