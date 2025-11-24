# Metaxy Integrations

## Metadata Stores

Learn more about metadata stores [here](../learn/metadata-stores.md).

<div class="grid cards" markdown>

<!-- dprint-ignore-start -->
-   :custom-deltalake:{ .lg .middle } __Delta Lake__

    ---

    :material-tag-outline: Storage Only

    Store metadata in Delta Lake format with support for local filesystem and remote object stores (S3, Azure, GCS).

    [:octicons-arrow-right-24: Integration docs](metadata-stores/storage/delta.md)

    [:octicons-arrow-right-24: API docs](../reference/api/metadata-stores/delta.md)

-   :custom-duckdb:{ .lg .middle } __DuckDB__

    ---

    :material-tag-outline: Database

    Store metadata in DuckDB with support for local files, in-memory databases, and MotherDuck cloud.

    [:octicons-arrow-right-24: Integration docs](metadata-stores/databases/duckdb.md)

    [:octicons-arrow-right-24: API docs](../reference/api/metadata-stores/ibis/duckdb.md)

-   :custom-clickhouse:{ .lg .middle } __ClickHouse__

    ---

    :material-tag-outline: Database

    Leverage ClickHouse for large metadata volumes and high-throughput setups.

    [:octicons-arrow-right-24: Integration docs](metadata-stores/databases/clickhouse.md)

    [:octicons-arrow-right-24: API docs](../reference/api/metadata-stores/ibis/clickhouse.md)
<!-- dprint-ignore-end -->

</div>

## Plugins

<div class="grid cards" markdown>

<!-- dprint-ignore-start -->
-   :custom-sqlalchemy:{ .lg .middle } __SQLAlchemy__

    ---

    :material-tag-outline: ORM • Database

    Retrieve SQLAlchemy URLs and `MetaData` for the current Metaxy project from Metaxy `MetadataStore` objects.

    [:octicons-arrow-right-24: Integration docs](plugins/sqlalchemy.md)

    [:octicons-arrow-right-24: API docs](../reference/api/ext/sqlalchemy.md)

-   :custom-sqlmodel:{ .lg .middle } __SQLModel__

    ---

    :material-tag-outline: ORM • Database

    Adds `SQLModel` capabilities to `metaxy.BaseFeature` class.

    [:octicons-arrow-right-24: Integration docs](plugins/sqlmodel.md)

    [:octicons-arrow-right-24: API docs](../reference/api/ext/sqlmodel.md)
<!-- dprint-ignore-end -->

</div>
