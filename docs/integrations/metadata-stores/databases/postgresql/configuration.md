---
title: "PostgreSQL Configuration"
description: "Configuration options for PostgreSQL metadata store."
---

# PostgreSQL Configuration

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.metadata_store.postgresql.PostgreSQLMetadataStoreConfig
    path_prefix: stores.dev.config
    header_level: 2
<!-- dprint-ignore-end -->

## PostgreSQL-Specific Options

### `auto_cast_struct_for_jsonb`

**Type**: `bool` | **Default**: `True`

Controls automatic conversion of Struct columns to JSONB format.

- **`True` (default)**: All DataFrame Struct columns (user-defined and Metaxy system columns) are automatically converted to JSONB on write and parsed back to Structs on read.

- **`False`**: Only Metaxy system columns (`metaxy_provenance_by_field`, `metaxy_data_version_by_field`) are converted. User-defined Struct columns are left as-is.

!!! tip "When to disable"

    Set `auto_cast_struct_for_jsonb=False` if you want full control over which user columns become JSONB in PostgreSQL. Metaxy system columns will always use JSONB since PostgreSQL lacks native struct support.
