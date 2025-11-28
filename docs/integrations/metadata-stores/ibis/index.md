# Ibis Integration

Metaxy uses [Ibis](https://ibis-project.org/) as a portable dataframe abstraction for SQL-based metadata stores. The [`IbisMetadataStore`][metaxy.metadata_store.ibis.IbisMetadataStore] is the base class for all SQL-backed stores.

## Available Backends

The following metadata stores are built on Ibis:

- [DuckDB](../databases/duckdb/index.md)
- [ClickHouse](../databases/clickhouse/index.md)
- [BigQuery](../databases/bigquery/index.md)

## Reference

- [API](api.md)
