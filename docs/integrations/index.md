# Metaxy Integrations

## Orchestration

<div class="grid cards" markdown>

<!-- dprint-ignore-start -->
-   :custom-dagster:{ .lg .middle } [**Dagster**](orchestration/dagster.md)

    ---

    :material-tag-outline: Orchestration • Data Platform

    Integrate Metaxy with Dagster using ConfigurableResource to manage metadata stores in your data pipelines.

    [:octicons-arrow-right-24: Integration docs](orchestration/dagster.md)

    [:octicons-arrow-right-24: API docs](../reference/api/ext/dagster.md)
<!-- dprint-ignore-end -->

</div>

## Metadata Stores

Learn more about metadata stores [here](../learn/metadata-stores.md).

<div class="grid cards" markdown>

<!-- dprint-ignore-start -->
-   :custom-deltalake:{ .lg .middle } [**Delta Lake**](metadata-stores/storage/delta.md)

    ---

    :material-tag-outline: Storage Only

    Store metadata in Delta Lake format with support for local filesystem and remote object stores (S3, Azure, GCS).

    [:octicons-arrow-right-24: Integration docs](metadata-stores/storage/delta.md)

    [:octicons-arrow-right-24: API docs](../reference/api/metadata-stores/delta.md)

-   :custom-duckdb:{ .lg .middle } [**DuckDB**](metadata-stores/databases/duckdb.md)

    ---

    :material-tag-outline: Database

    Store metadata in DuckDB with support for local files, in-memory databases, and MotherDuck cloud.

    [:octicons-arrow-right-24: Integration docs](metadata-stores/databases/duckdb.md)

    [:octicons-arrow-right-24: API docs](../reference/api/metadata-stores/ibis/duckdb.md)

-   :custom-clickhouse:{ .lg .middle } [**ClickHouse**](metadata-stores/databases/clickhouse.md)

    ---

    :material-tag-outline: Database

    Leverage ClickHouse for large metadata volumes and high-throughput setups.

    [:octicons-arrow-right-24: Integration docs](metadata-stores/databases/clickhouse.md)

    [:octicons-arrow-right-24: API docs](../reference/api/metadata-stores/ibis/clickhouse.md)

-   :custom-bigquery-blue:{ .lg .middle } [**BigQuery**](metadata-stores/databases/bigquery.md)

    ---

    :material-tag-outline: Database

    Use Google BigQuery as a scalable serverless metadata store on GCP.

    [:octicons-arrow-right-24: Integration docs](metadata-stores/databases/bigquery.md)

    [:octicons-arrow-right-24: API docs](../reference/api/metadata-stores/ibis/bigquery.md)
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

    [:octicons-arrow-right-24: API docs](../reference/api/ext/sqlalchemy.md)

-   :custom-sqlmodel:{ .lg .middle } [**SQLModel**](plugins/sqlmodel.md)

    ---

    :material-tag-outline: ORM • Database

    Adds `SQLModel` capabilities to `metaxy.BaseFeature` class.

    [:octicons-arrow-right-24: Integration docs](plugins/sqlmodel.md)

    [:octicons-arrow-right-24: API docs](../reference/api/ext/sqlmodel.md)
<!-- dprint-ignore-end -->

</div>
