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

---

<!-- dprint-ignore-start -->
::: metaxy.metadata_store.ibis
    options:
      members: false
      show_root_heading: true
      heading_level: 2

::: metaxy.metadata_store.ibis.IbisMetadataStore
    options:
      members: false
      heading_level: 3
<!-- dprint-ignore-end -->
