<!-- --8<-- [start:header] -->
# Changelog

All notable user-facing Metaxy changes are to be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), except for *experimental* features.

While Metaxy's core functionality and versioning engine are **_stable_** and quite complete, the CLI and some of the more advanced APIs and integrations are considered **_experimental_**.

!!! abstract

    **_stable_** features are guaranteed to follow SemVer and won't receive breaking changes in between minor releases. Such changes will be announced with a deprecation warning.
    A feature is considered *stable* if it's documented and doesn't have an `"Experimental"` badge.

    **_experimental_** features may be changed or removed at any time without deprecation warnings. They may be documented and in this case must display an `"Experimental"` warning badge:

    !!! warning "Experimental"
        This functionality is experimental.

<!-- --8<-- [end:header] -->
<!-- --8<-- [start:releases] -->

## v0.1.6 (18-03-2026)

### :bug: Bug Fixes

- **sqlalchemy**: the `SQLAlchemy` integration configuration (`[ext.sqlalchemy]` in `metaxy.toml`) has been removed. Please use the code interface instead. ([#1075](https://github.com/anam-org/metaxy/pull/1075) by [@danielgafni](https://github.com/danielgafni))
- **sqlmodel**: the `SQLModel` integration configuration (`[ext.sqlmodel]` in `metaxy.toml`) has been removed. Please use the code interface instead. ([#1074](https://github.com/anam-org/metaxy/pull/1074) by [@danielgafni](https://github.com/danielgafni))

### :book: Docs

- only display field name in configuration docs ([#1070](https://github.com/anam-org/metaxy/pull/1070) by [@danielgafni](https://github.com/danielgafni))

## v0.1.5 (18-03-2026)

### :sparkles: Features

- PostgreSQL metadata store ([#743](https://github.com/anam-org/metaxy/pull/743) by [@geoHeil](https://github.com/geoHeil))
- support Python 3.14 ([#1055](https://github.com/anam-org/metaxy/pull/1055) by [@geoHeil](https://github.com/geoHeil))
- Configuration inheritance is now supported via a new `extend` config field. ([#1060](https://github.com/anam-org/metaxy/pull/1060) by [@danielgafni](https://github.com/danielgafni))

### :bug: Bug Fixes

- display help when mx is run without arguments ([#1059](https://github.com/anam-org/metaxy/pull/1059) by [@AnkitSharma-29](https://github.com/AnkitSharma-29))

### :book: Docs

- refine PostgreSQL docs ([#1066](https://github.com/anam-org/metaxy/pull/1066) by [@danielgafni](https://github.com/danielgafni))
- fix ducklake example ([#1042](https://github.com/anam-org/metaxy/pull/1042) by [@geoHeil](https://github.com/geoHeil))

### :heart: New Contributors

- [@AnkitSharma-29](https://github.com/AnkitSharma-29) made their first contribution in [#1059](https://github.com/anam-org/metaxy/pull/1059)


## 0.1.4 (12-03-2026)

### :bug: Bug Fixes

- Fixed a bug with `metaxy push` not re-pushing feature definitions on changes reverting them to a previous version.

## 0.1.3 (11-03-2026)

### :sparkles: Features

- `metaxy lock` now avoids loading feature definitions from metaxy.project entrypoints of Python packages that aren't dependencies (direct or transitive) of the current Python package when creating the lock file.

### :wrench: Changed

- Renamed `metaxy graph history` CLI command to `metaxy history`

### :book: Docs

- Added a section on inheritance with Metaxy features
- Improved Dagster integration docs

## 0.1.2 (06-03-2026)

### :sparkles: Features

- A new `staleness_predicates` parameter has been added to [`resolve_update`][metaxy.MetadataStore.resolve_update]. It can be used to mark records as stale based on arbitrary Narwhals expressions, regardless of their Metaxy versions.
- `MetadataStore.write` now fills `metaxy_data_version` and `metaxy_data_version_by_field` values in place of `Null`s on per-row basis instead of checking whether the whole column presence.
- Added [Rebases](/guide/concepts/metadata-stores.md#rebasing-metadata-versions) functionality which allows backfilling metadata from historical feature versions. It's available via [CLI](/reference/cli.md#metaxy-metadata-rebase) or as [`MetadataStore.rebase`][metaxy.MetadataStore.rebase].

## 0.1.1 (27-02-2026)

### :sparkles: Features

- A new top-level [`extra_features`](/reference/configuration.md#metaxy.config.MetaxyConfig.extra_features) configuration option has been added. It can be used to register additional external feature definitions from the metadata store on the feature graph. Learn more [here](/guide/concepts/definitions/external-features.md/#loading-extra-features).
- [DuckLake](/integrations/metadata-stores/storage/ducklake.md) integration has been revamped, tested, and documented. Thanks [@geoHeil](https://github.com/geoHeil)!

### :bug: Bug Fixes

- Fixed `DuckDBMetadataStore` assuming `"community"` extension repo instead of `"core"` when loading extensions with unspecified repos
- Fixed `/metaxy` skill and `metaxy` MCP server not being properly installed by the Claude plugin
- (:boom: breaking) Fixed a bug where `MetadataStore` erroneously did not require being opened with `"w"` write mode before writing data
- Fixed a bug where `MetadataStore` would not properly handle nested context manager calls

## 0.1.0 (13-02-2026)

The first public Metaxy release :tada:!

This release should be considered an *alpha* release: it is ready for production use and has been dogfooded internally at [Anam.ai](https://anam.ai), but we don't have any community feedback yet.

### :sparkles: Features

- [Feature Definitions](/guide/concepts/definitions/features.md) and related models
- Metaxy [versioning engine](/guide/concepts/versioning.md) and [storage layout](/reference/system-columns.md)
- [MetadataStore API](/guide/concepts/metadata-stores.md)
- Integrations: [DeltaLake](/integrations/metadata-stores/storage/delta.md), [Dagster](/integrations/orchestration/dagster/index.md), [Ray](/integrations/compute/ray.md), [ClickHouse](/integrations/metadata-stores/databases/clickhouse.md), [MCP](/integrations/ai/mcp.md), [Claude](/integrations/ai/claude.md)

### :alembic: Experimental

- [CLI](/reference/cli.md)
- [`metaxy.lock`-based workflow](/guide/concepts/definitions/external-features.md/#metaxylock-file) for multi-environment setups
- Integrations: [DuckDB](/integrations/metadata-stores/databases/duckdb.md), [BigQuery](/integrations/metadata-stores/databases/bigquery.md), [LanceDB](/integrations/metadata-stores/databases/lancedb.md), [SQLAlchemy](/integrations/plugins/sqlalchemy.md), [SQLModel](/integrations/plugins/sqlmodel.md)

<!-- --8<-- [end:releases] -->
