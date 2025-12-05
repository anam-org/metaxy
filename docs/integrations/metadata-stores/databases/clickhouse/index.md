# Metaxy + ClickHouse

Metaxy implements [`ClickHouseMetadataStore`][metaxy.metadata_store.clickhouse.ClickHouseMetadataStore]. It uses [ClickHouse](https://clickhouse.com/) as metadata storage and versioning engine.

## Installation

```shell
pip install 'metaxy[clickhouse]'
```

## Struct Columns and JSON Storage

Metaxy uses struct columns (`metaxy_provenance_by_field`, `metaxy_data_version_by_field`) to track field-level versioning. In Python world this corresponds to `dict[str, str]`.

### How ClickHouse Handles Structs

ClickHouse offers multiple approaches for structured data:

| Type                                                                    | Description                       | Use Case                                             |
| ----------------------------------------------------------------------- | --------------------------------- | ---------------------------------------------------- |
| [`JSON`](https://clickhouse.com/docs/sql-reference/data-types/newjson)  | Native JSON with typed subcolumns | Improved query performance on JSON paths             |
| [`Map(K, V)`](https://clickhouse.com/docs/sql-reference/data-types/map) | Native key-value map              | When schema is `dict[K, V]`, even better performance |

### SQLAlchemy and Alembic Migrations

For better SQLAlchemy and Alembic migrations support, use the community [`clickhouse-sqlalchemy`](https://github.com/xzkostyan/clickhouse-sqlalchemy) driver:

```shell
pip install clickhouse-sqlalchemy
```

??? note "Alternative: ClickHouse Connect"

    Alternatively, use the official [`clickhouse-connect`](https://clickhouse.com/docs/integrations/python) driver.

!!! tip "Alembic Integration"

    See [Alembic setup guide](../../../plugins/sqlalchemy/index.md#alembic-integration) for additional instructions on how to use Alembic with Metaxy.

## Performance Optimization

!!! tip "Table Partitioning and Ordering"

    For optimal query performance, create your ClickHouse tables with:

    - **Partitioning**: Partition your tables!
    - **ORDER BY**: Use `(metaxy_feature_version, <id_columns>, metaxy_created_at)`

    ClickHouse doesn't have indexing, instead it relies on partitioning and ordering to optimize queries.

## Reference

- [Configuration](configuration.md)
- [API](api.md)
- [Introduction to ClickHouse](https://clickhouse.com/docs/intro)
- [ClickHouse Connect](https://clickhouse.com/docs/integrations/python)
