---
title: "PostgreSQL Metadata Store"
description: "PostgreSQL as a metadata store backend."
---

# Metaxy + PostgreSQL

Metaxy implements [`PostgreSQLMetadataStore`][metaxy.metadata_store.postgresql.PostgreSQLMetadataStore]. It uses [PostgreSQL](https://www.postgresql.org/) for metadata storage with [Polars](https://pola.rs/) for versioning compute.

**Hashing**: All provenance hashing is computed in Polars using xxHash64 (default) or other algorithms from the [polars-hash](https://github.com/ion-elgreco/polars-hash) plugin. Only the pre-computed hash results are stored in PostgreSQL. No PostgreSQL extensions are required.

!!! tip "Alembic Integration"

    See [Alembic setup guide](../../../plugins/sqlalchemy/index.md#alembic-integration) for instructions on using Alembic with Metaxy.

## Installation

```shell
pip install 'metaxy[postgres]'
```

## Metaxy's Versioning Struct Columns

Metaxy uses struct columns (`metaxy_provenance_by_field`, `metaxy_data_version_by_field`) to track field-level versioning. In Python world this corresponds to `dict[str, str]`. PostgreSQL doesn't have native struct types, so these columns are stored as JSONB.

### How PostgreSQL Handles Structs

PostgreSQL offers multiple approaches to represent structured data:

| Type                                                                     | Description               | Use Case                                           |
| ------------------------------------------------------------------------ | ------------------------- | -------------------------------------------------- |
| [`JSONB`](https://www.postgresql.org/docs/current/datatype-json.html)    | Binary JSON with indexing | **Recommended for Metaxy** because of dynamic keys |
| [`JSON`](https://www.postgresql.org/docs/current/datatype-json.html)     | Text-based JSON           | Less efficient than JSONB                          |
| [Composite Types](https://www.postgresql.org/docs/current/rowtypes.html) | User-defined row types    | More performant than JSONB but keys are static     |

!!! success "Recommended: `JSONB`"

    For Metaxy's `metaxy_provenance_by_field` and `metaxy_data_version_by_field` columns, use `JSONB`:

    - **No migrations required** when feature fields change

    - **GIN indexing support** for fast queries on field values

!!! warning "Automatic JSONB conversion"

    By default (`auto_cast_struct_for_jsonb=True`), Metaxy automatically converts **all Struct columns** to JSONB:

    - **Reading**: JSONB columns are converted to [Polars Structs][polars.datatypes.Struct] (e.g., `Struct[{"field_a": str, "field_b": str}]`)

    - **Writing**: Polars Structs are converted to JSONB format using `.struct.json_encode()`

    This includes both Metaxy system columns (`metaxy_provenance_by_field`, `metaxy_data_version_by_field`) and user-defined Struct columns. Set `auto_cast_struct_for_jsonb=False` to only convert Metaxy system columns.

## Performance Optimization

!!! tip "SQLModel/SQLAlchemy Integration"

    The [SQLModel integration](../../../plugins/sqlalchemy/index.md) handles table design automatically, including:

    - **Primary Key**: Composite key on ID columns + `metaxy_created_at` + `metaxy_feature_version`
    - **Indexes**: Composite index on the same columns

    Enable with `inject_primary_key=True` and `inject_index=True` in your configuration.

## Reference

- [Configuration](configuration.md)
- [API](api.md)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/current/)
- [JSONB Data Type](https://www.postgresql.org/docs/current/datatype-json.html)
