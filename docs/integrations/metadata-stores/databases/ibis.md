---
title: "Ibis Metadata Store"
description: "Ibis-based metadata store for SQL databases."
---

# Ibis Integration

Metaxy uses [Ibis](https://ibis-project.org/) as a portable dataframe abstraction for SQL-based metadata stores. The [`IbisMetadataStore`][metaxy.metadata_store.ibis.IbisMetadataStore] is the base class for all SQL-backed stores.

## Available Backends

The following metadata stores are built on Ibis:

- [DuckDB](duckdb.md)
- [ClickHouse](clickhouse.md)
- [BigQuery](bigquery.md)

## API

::: metaxy.metadata_store.ibis
options:
members: false

<!-- dprint-ignore-start -->
::: metaxy.metadata_store.ibis.IbisMetadataStore
    options:
      inherited_members: false
<!-- dprint-ignore-end -->
