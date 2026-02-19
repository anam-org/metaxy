---
title: "Integrations"
description: "Integrate Metaxy with other tooling, databases, external services, Python libraries, and more."
---

# Metaxy Integrations

## Orchestration

<div class="grid cards" markdown>

<!-- dprint-ignore-start -->
  - :custom-dagster:{ .lg .middle } [**Dagster**](orchestration/dagster/index.md)

    ---

    :material-tag-outline: Orchestration • Data Platform

    Seamlessly integrate Metaxy with [Dagster](https://dagster.io/) with the power of `@metaxify` and the `MetaxyIOManager`.

    !!! success "Recommended"

        Metaxy has been built with Dagster in mind. This integration is the best way to organize, materialize and observe multiple Metaxy features at scale.

    [:octicons-arrow-right-24: Integration docs](orchestration/dagster/index.md)

    [:octicons-arrow-right-24: API docs](orchestration/dagster/api.md)
<!-- dprint-ignore-end -->

</div>

## Metadata Stores

Learn more about metadata stores [here](../guide/concepts/metadata-stores.md).

<div class="annotate" markdown>
<div class="grid cards" markdown>

<!-- dprint-ignore-start -->

  - :custom-bigquery-blue:{ .lg .middle } [**BigQuery**](metadata-stores/databases/bigquery.md)

    ---

    :material-tag-outline: Database

    Use [Google BigQuery](https://cloud.google.com/bigquery) - scalable serverless analytical database on GCP.

    [:octicons-arrow-right-24: Integration docs](metadata-stores/databases/bigquery.md)

    [:octicons-arrow-right-24: API docs](metadata-stores/databases/bigquery.md)

  - :custom-clickhouse:{ .lg .middle } [**ClickHouse**](metadata-stores/databases/clickhouse.md)

    ---

    :material-tag-outline: Database

    Leverage the lightning-fast analytical [ClickHouse](https://clickhouse.com/) database for large metadata volume and high-throughput setups.

    !!! success "Recommended"

        Ideal for production.

    [:octicons-arrow-right-24: Integration docs](metadata-stores/databases/clickhouse.md)

    [:octicons-arrow-right-24: API docs](metadata-stores/databases/clickhouse.md)

  - :custom-deltalake:{ .lg .middle } [**Delta Lake**](metadata-stores/storage/delta.md)

    ---

    :material-tag-outline: Storage • Lakehouse

    Store metadata in [Delta Lake](https://delta.io/) format in local files or remote object stores (S3, GCS, and others). (1)

    !!! success "Recommended"

        Ideal for dev environments.

    [:octicons-arrow-right-24: Integration docs](metadata-stores/storage/delta.md)

    [:octicons-arrow-right-24: API docs](metadata-stores/storage/delta.md)

  - :custom-duckdb:{ .lg .middle } [**DuckDB**](metadata-stores/databases/duckdb.md)

    ---

    :material-tag-outline: Database • :material-tag-outline: Storage

    Use [DuckDB](https://duckdb.org/) - a fast analytical database with support for local and remote compute.

    !!! warning

        Local DuckDB is not recommended for production due to [parallel writes limitations](https://duckdb.org/docs/stable/connect/concurrency#writing-to-duckdb-from-multiple-processes).

    [:octicons-arrow-right-24: Integration docs](metadata-stores/databases/duckdb.md)

    [:octicons-arrow-right-24: API docs](metadata-stores/databases/duckdb.md)

  - :custom-duckdb:{ .lg .middle } [**DuckLake**](metadata-stores/storage/ducklake.md)

    ---

    :material-tag-outline: Storage • Lakehouse

    Use the very performant [DuckLake](https://ducklake.select/) lakehouse format for storing Metaxy metadata.

    [:octicons-arrow-right-24: Integration docs](metadata-stores/storage/ducklake.md)

  - :custom-lancedb:{ .lg .middle } [**LanceDB**](metadata-stores/databases/lancedb.md)

    ---

    :material-tag-outline: Database • :material-tag-outline: Storage

    Use the multimodal [LanceDB](https://lancedb.com/) database or [Lance](https://lancedb.com/docs/overview/lance/) storage format. (2)

    [:octicons-arrow-right-24: Integration docs](metadata-stores/databases/lancedb.md)

    [:octicons-arrow-right-24: API docs](metadata-stores/databases/lancedb.md)
<!-- dprint-ignore-end -->

</div>
</div>

1. uses a local versioning engine implemented in [Polars](https://docs.pola.rs/) and [`polars-hash`](https://github.com/ion-elgreco/polars-hash)

2. uses a local versioning engine implemented in [Polars](https://docs.pola.rs/) and [`polars-hash`](https://github.com/ion-elgreco/polars-hash)

## Compute

<div class="grid cards" markdown>

<!-- dprint-ignore-start -->
-   :custom-ray:{ .lg .middle } [**Ray**](compute/ray.md)

    ---

    :material-tag-outline: Compute • Distributed

    Use Metaxy with [Ray](https://ray.io/) for distributed computing workloads.

    [:octicons-arrow-right-24: Integration docs](compute/ray.md)
<!-- dprint-ignore-end -->

</div>

## Plugins

<div class="grid cards" markdown>

<!-- dprint-ignore-start -->
-   :custom-sqlalchemy:{ .lg .middle } [**SQLAlchemy**](plugins/sqlalchemy.md)

    ---

    :material-tag-outline: ORM • Database

    Retrieve SQLAlchemy URLs and `MetaData` for the current Metaxy project from Metaxy `MetadataStore` objects.

    [:octicons-arrow-right-24: Integration docs](plugins/sqlalchemy.md)

    [:octicons-arrow-right-24: API docs](plugins/sqlalchemy.md)

-   :custom-sqlmodel:{ .lg .middle } [**SQLModel**](plugins/sqlmodel.md)

    ---

    :material-tag-outline: ORM • Database

    Adds `SQLModel` capabilities to `metaxy.BaseFeature` class.

    [:octicons-arrow-right-24: Integration docs](plugins/sqlmodel.md)

    [:octicons-arrow-right-24: API docs](plugins/sqlmodel.md)
<!-- dprint-ignore-end -->

</div>

## AI

<div class="grid cards" markdown>

<!-- dprint-ignore-start -->
-   :custom-claude:{ .lg .middle } [**Claude Code**](ai/claude.md)

    ---

    :material-tag-outline: AI • LLM

    Use Metaxy with [Claude Code](https://claude.ai/claude-code) through the official plugin, providing the `/metaxy` skill and MCP tools.

    [:octicons-arrow-right-24: Integration docs](ai/claude.md)

-   :custom-mcp:{ .lg .middle } [**MCP Server**](ai/mcp.md)

    ---

    :material-tag-outline: AI • LLM

    Expose Metaxy's feature graph and metadata store operations to AI assistants via the [Model Context Protocol](https://modelcontextprotocol.io/).

    [:octicons-arrow-right-24: Integration docs](ai/mcp.md)
<!-- dprint-ignore-end -->

</div>
