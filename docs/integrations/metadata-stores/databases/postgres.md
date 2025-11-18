# PostgresMetadataStore

Metaxy implements [`PostgresMetadataStore`][metaxy.metadata_store.postgres.PostgresMetadataStore].
It uses [PostgreSQL](https://www.postgresql.org/) as metadata storage and versioning engine.

## Installation

```shell
pip install 'metaxy[postgres]'
```

## Configuration

Pass a PostgreSQL connection string instead of individual parameters if you prefer DSN-style configuration.
Enabling `pgcrypto` is optional but allows the store to use PostgreSQL-native hashing when `hash_algorithm` is set to `SHA256`.
