# Ibis Integration

Metaxy uses [Ibis](https://ibis-project.org/) as a portable dataframe abstraction for SQL-based metadata stores. The [`IbisMetadataStore`][metaxy.metadata_store.ibis.IbisMetadataStore] is the base class for all SQL-backed stores.

## Available Backends

The following metadata stores are built on Ibis:

- [DuckDB](../duckdb/index.md)
- [ClickHouse](../clickhouse/index.md)
- [BigQuery](../bigquery/index.md)

## Reference

- [API](api.md)
