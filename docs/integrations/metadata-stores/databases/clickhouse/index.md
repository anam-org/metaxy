# Metaxy + ClickHouse

Metaxy implements [`ClickHouseMetadataStore`][metaxy.metadata_store.clickhouse.ClickHouseMetadataStore]. It uses [ClickHouse](https://clickhouse.com/) as metadata storage and versioning engine.

## Installation

```shell
pip install 'metaxy[clickhouse]'
```

## Metaxy's Versioning Struct Columns

Metaxy uses struct columns (`metaxy_provenance_by_field`, `metaxy_data_version_by_field`) to track field-level versioning. In Python world this corresponds to `dict[str, str]`. In ClickHouse, there are several options to represent these columns.

### How ClickHouse Handles Structs

ClickHouse offers multiple approaches to represent Metaxy's structured versioning columns:

| Type                                                                                                                | Description                       | Use Case                                                                   |
| ------------------------------------------------------------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------- |
| [`Map(String, String)`](https://clickhouse.com/docs/sql-reference/data-types/map)                                   | Native key-value map              | **Recommended for Metaxy** because of dynamic keys                         |
| [`JSON`](https://clickhouse.com/docs/sql-reference/data-types/newjson)                                              | Native JSON with typed subcolumns | Less performant than `Map(String, String)` but more flexible than `Nested` |
| [`Nested(field_1 String, ...)`](https://clickhouse.com/docs/sql-reference/data-types/nested-data-structures/nested) | Static struct with named fields   | More performant than `Map(String, String)` but keys are static             |

!!! success "Recommended: `Map(String, String)`"

    For Metaxy's `metaxy_provenance_by_field` and `metaxy_data_version_by_field` columns, use `Map(String, String)`:

    - **No migrations required** when feature fields change

    - **Good performance** for key-value lookups

### SQLAlchemy and Alembic Migrations

For SQLAlchemy and Alembic migrations support, use the [`clickhouse-sqlalchemy`](https://github.com/xzkostyan/clickhouse-sqlalchemy) driver with the **native protocol**:

```shell
pip install clickhouse-sqlalchemy
```

!!! warning "Use Native Clickhouse Protocol"

    The HTTP protocol has [limited reflection support](https://github.com/xzkostyan/clickhouse-sqlalchemy/issues/15). Always use the native protocol (`clickhouse+native://`) for full SQLAlchemy/Alembic compatibility:

    ```python
    connection_string = "clickhouse+native://user:pass@localhost:9000/default"
    ```

    The [`ClickHouseMetadataStore.sqlalchemy_url`][metaxy.metadata_store.clickhouse.ClickHouseMetadataStore.sqlalchemy_url] property is tweaked to return the native connection string variant.

??? note "Alternative: ClickHouse Connect"

    Alternatively, use the official [`clickhouse-connect`](https://clickhouse.com/docs/integrations/python) driver.

!!! tip "Alembic Integration"

    See [Alembic setup guide](../../../plugins/sqlalchemy/index.md#alembic-integration) for additional instructions on how to use Alembic with Metaxy.

## Performance Optimization

!!! tip "Table Design"

    For optimal query performance, create your ClickHouse tables with:

    - **Partitioning**: Partition your tables!
    - **Primary Key**: It's probably a good idea to use `(metaxy_feature_version, <id_columns>, metaxy_created_at)`

## Reference

- [Configuration](configuration.md)
- [API](api.md)
- [Introduction to ClickHouse](https://clickhouse.com/docs/intro)
- [ClickHouse Connect](https://clickhouse.com/docs/integrations/python)
