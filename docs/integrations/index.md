# Metaxy Integrations

## Orchestration

<div class="grid cards" markdown>

<!-- dprint-ignore-start -->
-   :custom-dagster:{ .lg .middle } [**Dagster**](orchestration/dagster/index.md)

    ---

    :material-tag-outline: Orchestration • Data Platform

    Integrate Metaxy with Dagster using ConfigurableResource to manage metadata stores in your data pipelines.

    [:octicons-arrow-right-24: Integration docs](orchestration/dagster/index.md)

    [:octicons-arrow-right-24: API docs](orchestration/dagster/api.md)
<!-- dprint-ignore-end -->

</div>

## Metadata Stores

Learn more about metadata stores [here](../guide/learn/metadata-stores.md).

<div class="grid cards" markdown>

<!-- dprint-ignore-start -->
-   :custom-deltalake:{ .lg .middle } [**Delta Lake**](metadata-stores/storage/delta/index.md)

    ---

    :material-tag-outline: Storage Only

    Store metadata in Delta Lake format with support for local filesystem and remote object stores (S3, Azure, GCS).

    [:octicons-arrow-right-24: Integration docs](metadata-stores/storage/delta/index.md)

    [:octicons-arrow-right-24: API docs](metadata-stores/storage/delta/api.md)

-   :custom-lancedb:{ .lg .middle } [**LanceDB**](metadata-stores/databases/lancedb/index.md)

    ---

    :material-tag-outline: Database

    Columnar database optimized for vector search and multimodal data with embedded (local) and external (object store or LanceDB Cloud) deployments.

    [:octicons-arrow-right-24: Integration docs](metadata-stores/databases/lancedb/index.md)

    [:octicons-arrow-right-24: API docs](metadata-stores/databases/lancedb/api.md)

-   :custom-duckdb:{ .lg .middle } [**DuckDB**](metadata-stores/databases/duckdb/index.md)

    ---

    :material-tag-outline: Database

    Store metadata in DuckDB with support for local files, in-memory databases, and MotherDuck cloud.

    [:octicons-arrow-right-24: Integration docs](metadata-stores/databases/duckdb/index.md)

    [:octicons-arrow-right-24: API docs](metadata-stores/databases/duckdb/api.md)

-   :custom-clickhouse:{ .lg .middle } [**ClickHouse**](metadata-stores/databases/clickhouse/index.md)

    ---

    :material-tag-outline: Database

    Leverage ClickHouse for large metadata volumes and high-throughput setups.

    [:octicons-arrow-right-24: Integration docs](metadata-stores/databases/clickhouse/index.md)

    [:octicons-arrow-right-24: API docs](metadata-stores/databases/clickhouse/api.md)

-   :custom-bigquery-blue:{ .lg .middle } [**BigQuery**](metadata-stores/databases/bigquery/index.md)

    ---

    :material-tag-outline: Database

    Use Google BigQuery as a scalable serverless metadata store on GCP.

    [:octicons-arrow-right-24: Integration docs](metadata-stores/databases/bigquery/index.md)

    [:octicons-arrow-right-24: API docs](metadata-stores/databases/bigquery/api.md)
<!-- dprint-ignore-end -->

</div>

## Plugins

<div class="grid cards" markdown>

<!-- dprint-ignore-start -->
-   :custom-sqlalchemy:{ .lg .middle } [**SQLAlchemy**](plugins/sqlalchemy/index.md)

    ---

    :material-tag-outline: ORM • Database

    Retrieve SQLAlchemy URLs and `MetaData` for the current Metaxy project from Metaxy `MetadataStore` objects.

    [:octicons-arrow-right-24: Integration docs](plugins/sqlalchemy/index.md)

    [:octicons-arrow-right-24: API docs](plugins/sqlalchemy/api.md)

-   :custom-sqlmodel:{ .lg .middle } [**SQLModel**](plugins/sqlmodel/index.md)

    ---

    :material-tag-outline: ORM • Database

    Adds `SQLModel` capabilities to `metaxy.BaseFeature` class.

    [:octicons-arrow-right-24: Integration docs](plugins/sqlmodel/index.md)

    [:octicons-arrow-right-24: API docs](plugins/sqlmodel/api.md)
<!-- dprint-ignore-end -->

</div>
