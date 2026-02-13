<!-- --8<-- [start:header] -->
# Changelog

All notable user-facing Metaxy changes are to be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), except for *experimental* features.

While Metaxy's core functionality and versioning engine is **_stable_** and quite complete, the CLI and some of the more advanced APIs and integrations are considered **_experimental_**.

!!! abstract

    **_stable_** features are guaranteed to follow SemVer and won't receive breaking changes in between minor releases. Such changes will be announced with a deprecation warning.
    A feature is considered *stable* if it's documented and doesn't have an `"Experimental"` badge.

    **_experimental_** features may be changed or removed at any time without deprecation warnings. They may be documented and in this case must display an `"Experimental"` warning badge:

    !!! warning "Experimental"
        This functionality is experimental.

<!-- --8<-- [end:header] -->

<!-- --8<-- [start:releases] -->

## 0.1.0

The first public Metaxy release :tada:!

This release should be considered an *alpha* release: it is ready for production use and has dogfooded at [Anam.ai](https://anam.ai) internally, but we don't have any community feedback yet.

### Added

- [Feature Definitions](/guide/concepts/definitions/features.md) and related models

- Metaxy [versioning engine](/guide/concepts/versioning.md) and [storage layout](/reference/system-columns.md)

- [MetadataStore API](/guide/concepts/metadata-stores.md)

- Integrations: [DeltaLake](/integrations/metadata-stores/storage/delta.md), [Dagster](/integrations/orchestration/dagster/index.md), [Ray](/integrations/compute/ray.md), [ClickHouse](/integrations/metadata-stores/databases/clickhouse.md), [MCP](/integrations/ai/mcp.md), [Claude](/integrations/ai/claude.md)

### Experimental

- [CLI](/reference/cli.md)

- [`metaxy.lock`-based workflow](/guide/concepts/definitions/external-features.md/#metaxylock-file) for multi-environment setups

- Integrations: [DuckDB](/integrations/metadata-stores/databases/duckdb.md), [BigQuery](/integrations/metadata-stores/databases/bigquery.md), [LanceDB](/integrations/metadata-stores/databases/lancedb.md), [SQLAlchemy](/integrations/plugins/sqlalchemy.md), [SQLModel](/integrations/plugins/sqlmodel.md)

<!-- --8<-- [end:releases] -->
