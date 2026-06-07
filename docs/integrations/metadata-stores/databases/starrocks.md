---
title: "Metaxy + StarRocks"
description: "Learn how to use StarRocks as a Metaxy metadata store."
---

# Metaxy + StarRocks

!!! warning "Experimental"

    This functionality is experimental.

[StarRocks](https://www.starrocks.io/) is an MPP analytical database. Metaxy connects to StarRocks through its MySQL-compatible query port, normally `9030`, and provides two StarRocks store modes.

| Mode                  | Store                                                                          | Versioning                 | Use when                                                                 |
| --------------------- | ------------------------------------------------------------------------------ | -------------------------- | ------------------------------------------------------------------------ |
| Normal MySQL mode     | [`StarRocksMySQLMetadataStore`][metaxy.ext.starrocks.StarRocksMySQLMetadataStore] | Polars                     | You want the compatibility baseline while storing metadata in StarRocks. |
| Optimized native mode | [`StarRocksMetadataStore`][metaxy.ext.starrocks.StarRocksMetadataStore]           | StarRocks-native SQL/Ibis  | You want StarRocks SQL pushdown and latest-row acceleration.             |

Both modes use StarRocks table DDL for automatically created feature tables. The optimized native mode additionally maintains StarRocks Aggregate-table projections for fast latest-row reads.

## Installation

```shell
pip install 'metaxy[starrocks]'
```

## Store Modes

### Normal MySQL Mode

[`StarRocksMySQLMetadataStore`][metaxy.ext.starrocks.StarRocksMySQLMetadataStore] stores feature history in StarRocks but uses the local Polars versioning engine. It is the simpler compatibility path and is useful for validating StarRocks storage behavior against Metaxy's Polars reference implementation.

This mode does not maintain the `__metaxy_latest` Aggregate tables, even if `enable_latest_aggregate_table` is set in config.

```toml
[stores.dev]
type = "metaxy.ext.starrocks.StarRocksMySQLMetadataStore"

[stores.dev.config]
connection_string = "mysql://root@127.0.0.1:9030/metaxy"
replication_num = 1
```

### Optimized Native Mode

[`StarRocksMetadataStore`][metaxy.ext.starrocks.StarRocksMetadataStore] pushes versioning into StarRocks through a StarRocks-native versioning engine. It uses StarRocks SQL functions for hashing, JSON extraction, ordered string aggregation, and latest-row selection.

By default, this mode also maintains one companion Aggregate table per feature table:

```text
<feature_table>__metaxy_latest
```

The companion table is keyed by the feature id columns plus `metaxy_feature_version`, and stores `metaxy_latest_lifecycle_at` as a `DATETIME MAX` value. On latest-row reads for a single feature version, Metaxy joins the canonical feature table to this projection before applying the final latest-row tie breaker. If the projection is disabled, missing, or not applicable to the read, Metaxy falls back to the standard native row-number latest selection.

The canonical feature table remains the source of truth. Aggregate tables are derived projections used only for acceleration.

```toml
[stores.dev]
type = "metaxy.ext.starrocks.StarRocksMetadataStore"

[stores.dev.config]
connection_string = "mysql://root@127.0.0.1:9030/metaxy"
replication_num = 1
enable_latest_aggregate_table = true
```

## Connection Configuration

Use either `connection_string` or `connection_params`. StarRocks' MySQL-compatible query port is usually `9030`.

```toml
[stores.dev.config]
connection_string = "mysql://root@127.0.0.1:9030/metaxy"
```

or:

```toml
[stores.dev.config]
connection_params = { host = "127.0.0.1", port = 9030, database = "metaxy", user = "root" }
```

For single-node development and Nix smoke tests, keep `replication_num = 1`. Leave `buckets` unset unless you want to choose the StarRocks bucket count explicitly.

## Table Layout

StarRocks requires table-type and distribution clauses, so Metaxy uses raw StarRocks DDL when auto-creating feature tables:

- canonical feature tables use `DUPLICATE KEY(<id columns>)`
- tables are distributed with `DISTRIBUTED BY HASH(<id columns>)`
- `replication_num` is emitted as a table property when configured
- `buckets` is emitted only when configured, otherwise StarRocks chooses automatically

Metaxy's by-field system columns, `metaxy_provenance_by_field` and `metaxy_data_version_by_field`, are stored as JSON strings. This keeps MySQL-wire insertion reliable while allowing native StarRocks JSON extraction in optimized mode.

## Hashing

The optimized native mode currently supports `xxh3_64`, `md5`, and `sha256`. Its default is `xxh3_64`, which maps to StarRocks' `xx_hash3_64` function.

`xxh3_64` is not the same algorithm as classic `xxhash64`. The native store therefore does not map Metaxy's classic `xxhash64` option to StarRocks' `xx_hash3_64`.

## API Reference

<!-- dprint-ignore-start -->
::: metaxy.ext.starrocks
    options:
      members: false
      show_root_heading: true
      heading_level: 2

::: metaxy.ext.starrocks.StarRocksMySQLMetadataStore
    options:
      members: false
      heading_level: 3

::: metaxy.ext.starrocks.StarRocksMetadataStore
    options:
      members: false
      heading_level: 3
<!-- dprint-ignore-end -->

## Configuration

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.ext.starrocks.StarRocksMetadataStoreConfig
    path_prefix: stores.dev.config
    header_level: 3
<!-- dprint-ignore-end -->
