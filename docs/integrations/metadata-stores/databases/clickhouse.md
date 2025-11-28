# Metaxy + ClickHouse

Metaxy implements [`ClickHouseMetadataStore`][metaxy.metadata_store.clickhouse.ClickHouseMetadataStore]. It uses [ClickHouse](https://clickhouse.com/) as metadata storage and versioning engine.

See [Configuration Reference](../../../reference/api/metadata-stores/ibis/clickhouse.md#configuration) for all available options.

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

### Alembic Migrations

For Alembic migrations, use [`clickhouse-connect`](https://clickhouse.com/docs/integrations/python):

```shell
pip install clickhouse-connect
```

??? note "Alternative Community Driver"

    Alternatively, use the community [`clickhouse-sqlalchemy`](https://github.com/xzkostyan/clickhouse-sqlalchemy) driver.

!!! tip "Alembic Integration"

    See [Alembic setup guide](../../plugins/sqlalchemy.md#alembic-integration) for additional instructions on how to use Alembic with Metaxy.

# Reference

- [Configuration Reference](../../../reference/api/metadata-stores/ibis/clickhouse.md#configuration)
- [API Reference][metaxy.metadata_store.clickhouse.ClickHouseMetadataStore]
- [Introduction to ClickHouse](https://clickhouse.com/docs/intro)
- [ClickHouse Connect](https://clickhouse.com/docs/integrations/python)
