---
title: "PostgreSQL Metadata Store"
description: "PostgreSQL as a metadata store backend."
---

# Metaxy + PostgreSQL

Metaxy implements [`PostgreSQLMetadataStore`][metaxy.metadata_store.postgresql.PostgreSQLMetadataStore]. It uses [PostgreSQL](https://www.postgresql.org/) for metadata storage with [Polars](https://pola.rs/) for versioning compute.

**Hashing**: All provenance hashing is computed in Polars using xxHash32 (default) or other algorithms from the [polars-hash](https://github.com/ion-elgreco/polars-hash) plugin. Only the pre-computed hash results are stored in PostgreSQL. No PostgreSQL extensions are required.

!!! tip "Alembic Integration"

    See [Alembic setup guide](../../plugins/sqlalchemy.md#alembic-integration) for instructions on using Alembic with Metaxy.

## Installation

```shell
pip install 'metaxy[postgres]'
```

## Metaxy's Versioning Struct Columns

Metaxy uses struct columns (`metaxy_provenance_by_field`, `metaxy_data_version_by_field`) to track field-level versioning. In Python world this corresponds to `dict[str, str]`. PostgreSQL doesn't have native struct types, so these columns should be stored as `JSONB` (recommended), though actual storage type depends on your table schema.

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

!!! warning "Automatic Struct JSON encoding"

    By default (`auto_cast_struct_for_jsonb=True`), Metaxy automatically JSON-encodes **all Struct columns** on write:

    - **Reading**: JSON/JSONB columns are converted to [Polars Structs][polars.datatypes.Struct] (e.g., `Struct[{"field_a": str, "field_b": str}]`)

    - **Writing**: Polars Structs are serialized to JSON strings using `.struct.json_encode()`

    PostgreSQL storage type is determined by your destination schema/casting:
    columns typed as `JSONB` store binary JSON, while `TEXT` columns store plain strings.
    This behavior includes Metaxy system columns (`metaxy_provenance_by_field`, `metaxy_data_version_by_field`) and user-defined Struct columns. Set `auto_cast_struct_for_jsonb=False` to only auto-encode Metaxy system columns.
    When `auto_create_tables=True`, PostgreSQL tables created by Metaxy use `JSONB`
    for Struct columns that Metaxy auto-encodes.

## Performance Optimization

!!! tip "SQLModel/SQLAlchemy Integration"

    The [SQLModel integration](../../plugins/sqlalchemy.md) handles table design automatically, including:

    - **Primary Key**: Composite key on ID columns + `metaxy_created_at` + `metaxy_feature_version`
    - **Indexes**: Composite index on the same columns

    Enable with `inject_primary_key=True` and `inject_index=True` in your configuration.

---

<!-- dprint-ignore-start -->
::: metaxy.metadata_store.postgresql
    options:
      members: false
      show_root_heading: true
      heading_level: 2

::: metaxy.metadata_store.postgresql.PostgreSQLMetadataStore
    options:
      inherited_members: false
      heading_level: 3

::: metaxy.metadata_store.postgresql.PostgreSQLMetadataStoreConfig
    options:
      inherited_members: false
      heading_level: 3
<!-- dprint-ignore-end -->

## Configuration

### `auto_cast_struct_for_jsonb`

**Type**: `bool` | **Default**: `True`

Controls automatic Struct JSON encoding on write and Struct parsing on read.

!!! note "Read Execution Model"
    PostgreSQL read-path JSON decoding is eager: Metaxy executes the SQL query,
    materializes the result to Polars, decodes JSON columns, then returns a
    LazyFrame over that in-memory snapshot. The returned LazyFrame is not a
    deferred database query.

- **`True` (default)**:
    - On **write**: All DataFrame Struct columns (user-defined and Metaxy system columns) are JSON-encoded (via `.struct.json_encode()`).
    - On **read**:
        - Metaxy system Struct columns (`metaxy_provenance_by_field`, `metaxy_data_version_by_field`) are always parsed back to Structs.
        - User-defined `JSON`/`JSONB` columns are parse candidates and JSON object payloads are decoded back to Structs.
        - User-defined `TEXT` columns are kept as strings, even if they contain JSON-looking payloads.

- **`False`**:
    - Only Metaxy system Struct columns (`metaxy_provenance_by_field`, `metaxy_data_version_by_field`) are automatically JSON-encoded on write and parsed back to Structs on read.
    - User-defined Struct columns are left as-is on both write and read.

!!! tip "When to disable"

    Set `auto_cast_struct_for_jsonb=False` if you want full control over which user columns are JSON-encoded/decoded. Final PostgreSQL storage (`TEXT` vs `JSON`/`JSONB`) and whether user Struct columns are parsed back on read depend on your table schema and any explicit SQL casts.

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.metadata_store.postgresql.PostgreSQLMetadataStoreConfig
    path_prefix: stores.dev.config
    header_level: 2
<!-- dprint-ignore-end -->
