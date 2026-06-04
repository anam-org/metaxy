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

## v0.1.10 (09-04-2026)

### :sparkles: Features

#### Map datatype

- use Arrow conversion utils from `polars-map` ([#1175](https://github.com/anam-org/metaxy/pull/1175) by [@danielgafni](https://github.com/danielgafni))


### :bug: Bug Fixes

- conditionally import DeltaLake and Iceberg stores in `metaxy.ext.polars` ([#1180](https://github.com/anam-org/metaxy/pull/1180) by [@geoHeil](https://github.com/geoHeil))


### :book: Docs

- fix "outdated version" banner in `main` docs version ([#1172](https://github.com/anam-org/metaxy/pull/1172) by [@danielgafni](https://github.com/danielgafni))


### :hammer_and_wrench: Other Improvements

- move Ruff config to `ruff.toml` ([#1177](https://github.com/anam-org/metaxy/pull/1177) by [@danielgafni](https://github.com/danielgafni))
- pin write-read Arrow schema for `MapDtype` tests ([#1176](https://github.com/anam-org/metaxy/pull/1176) by [@danielgafni](https://github.com/danielgafni))
- fix changelog for #1173 ([#1179](https://github.com/anam-org/metaxy/pull/1179) by [@danielgafni](https://github.com/danielgafni))
- update devenv.lock ([#1173](https://github.com/anam-org/metaxy/pull/1173) by [@geoHeil](https://github.com/geoHeil))

#### Deltalake
- the `metaxy.ext.polars.DeltaMetadataStore` import path has been deprecated in favor of `metaxy.ext.polars.handlers.delta.DeltaMetadataStore` ([#1181](https://github.com/anam-org/metaxy/pull/1181) by [@danielgafni](https://github.com/danielgafni))

#### Iceberg
- the `metaxy.ext.iceberg.IcebergMetadataStore` import path has been deprecated in favor of `metaxy.ext.polars.handlers.iceberg.IcebergMetadataStore` ([#1181](https://github.com/anam-org/metaxy/pull/1181) by [@danielgafni](https://github.com/danielgafni))

#### Publications

- paper SAO Workshop ([#1174](https://github.com/anam-org/metaxy/pull/1174) by [@geoHeil](https://github.com/geoHeil))


## v0.1.9 (03-04-2026)

### :wastebasket: Deprecated

- `MetadataStore` implementations have been moved from `metaxy.ext.metadata_stores.<store>` to `metaxy.ext.<store>`. Old import paths are deprecated and will be removed in `0.2.0`.

- `HashAlgorithm` has been moved from `metaxy.versioning.types` to `metaxy._hashing` (is available as `metaxy.HashAlgorithm`). The old import path is deprecated and will be removed in `0.2.0`.


### :sparkles: Features

#### Map datatype

- [`narwhals-map`](https://pypi.org/project/narwhals-map/) is now used for `Map` datatype operations in Narwhals. This changes the Narwhals datatype from `Unknown` to `narwhals_map.Map` and exposes `nw.Expr.map` namespace for these columns. ([#1167](https://github.com/anam-org/metaxy/pull/1167) by [@danielgafni](https://github.com/danielgafni))


### :book: Docs

- default to `stable` docs version and updatge all links ([#1156](https://github.com/anam-org/metaxy/pull/1156) by [@danielgafni](https://github.com/danielgafni))

#### Map datatype

- document `narwhals-map` usage for `Map` datatype ([#1171](https://github.com/anam-org/metaxy/pull/1171) by [@danielgafni](https://github.com/danielgafni))


### :hammer_and_wrench: Other Improvements

- release v0.1.9 ([#1170](https://github.com/anam-org/metaxy/pull/1170) by [@danielgafni](https://github.com/danielgafni))
- fix random old analysis script ([#1168](https://github.com/anam-org/metaxy/pull/1168) by [@danielgafni](https://github.com/danielgafni))
- improve internal dependency structure ([#1166](https://github.com/anam-org/metaxy/pull/1166) by [@danielgafni](https://github.com/danielgafni))
- remove a few circular deps ([#1164](https://github.com/anam-org/metaxy/pull/1164) by [@danielgafni](https://github.com/danielgafni))
- introduce Tach layers ([#1163](https://github.com/anam-org/metaxy/pull/1163) by [@danielgafni](https://github.com/danielgafni))
- fix `source_roots` and remove utility modules in  `tach.toml` ([#1162](https://github.com/anam-org/metaxy/pull/1162) by [@danielgafni](https://github.com/danielgafni))
- run all integration tests in `main` ([#1161](https://github.com/anam-org/metaxy/pull/1161) by [@danielgafni](https://github.com/danielgafni))
- fix duckdb tests referring old paths ([#1160](https://github.com/anam-org/metaxy/pull/1160) by [@danielgafni](https://github.com/danielgafni))
- move `PolarsVersioningEngine` to `metaxy.ext.polars.versioning` ([#1159](https://github.com/anam-org/metaxy/pull/1159) by [@danielgafni](https://github.com/danielgafni))
- update old extension paths ([#1158](https://github.com/anam-org/metaxy/pull/1158) by [@danielgafni](https://github.com/danielgafni))
- move ibis code to `metaxy.ext.ibis` ([#1157](https://github.com/anam-org/metaxy/pull/1157) by [@danielgafni](https://github.com/danielgafni))
- migrate tests to use new metadata store imports ([#1155](https://github.com/anam-org/metaxy/pull/1155) by [@danielgafni](https://github.com/danielgafni))
- Metadata stores have been moved from `metaxy.ext.metadata_stores.<store>` to `metaxy.ext.<store>`. DeltaLake and Iceberg stores are additionally nested under `metaxy.ext.polars.handlers.`. Old locations have been kept for backwards compatibility but will be removed in `0.2.0`. ([#1153](https://github.com/anam-org/metaxy/pull/1153) by [@danielgafni](https://github.com/danielgafni))
- update tach modules for integrations ([#1152](https://github.com/anam-org/metaxy/pull/1152) by [@danielgafni](https://github.com/danielgafni))
- fix coverage for `main` branch ([#1151](https://github.com/anam-org/metaxy/pull/1151) by [@danielgafni](https://github.com/danielgafni))
- fix coverage for release PRs by testing `metaxy.__version__` ([#1150](https://github.com/anam-org/metaxy/pull/1150) by [@danielgafni](https://github.com/danielgafni))

#### Claude

- add Tach skill ([#1165](https://github.com/anam-org/metaxy/pull/1165) by [@danielgafni](https://github.com/danielgafni))


## v0.1.8 (01-04-2026)

### :sparkles: Features

#### Claude

- more info on working with `Map` columns ([#1141](https://github.com/anam-org/metaxy/pull/1141) by [@danielgafni](https://github.com/danielgafni))

#### Cli

- more detailed and user-friendly output for `metaxy lock` ([#1135](https://github.com/anam-org/metaxy/pull/1135) by [@danielgafni](https://github.com/danielgafni))

#### Ray

- support `Map` datatype ([#1142](https://github.com/anam-org/metaxy/pull/1142) by [@danielgafni](https://github.com/danielgafni))


### :bug: Bug Fixes

- remove missing lock file warning when `metaxy_lock_path` is not configured ([#1143](https://github.com/anam-org/metaxy/pull/1143) by [@danielgafni](https://github.com/danielgafni))
- improve `Map` datatype support in `MetadataStore.resolve_update` ([#1139](https://github.com/anam-org/metaxy/pull/1139) by [@danielgafni](https://github.com/danielgafni))


### :book: Docs

- more info on working with `Map` columns ([#1140](https://github.com/anam-org/metaxy/pull/1140) by [@danielgafni](https://github.com/danielgafni))

#### Sqlmode

- add docstrings for `SQLModelFeatureConfig` ([#1127](https://github.com/anam-org/metaxy/pull/1127) by [@danielgafni](https://github.com/danielgafni))


### :hammer_and_wrench: Other Improvements

- release v0.1.8 ([#1147](https://github.com/anam-org/metaxy/pull/1147) by [@danielgafni](https://github.com/danielgafni))
- fix tach setup in `main` branch ([#1149](https://github.com/anam-org/metaxy/pull/1149) by [@danielgafni](https://github.com/danielgafni))
- enable Tach pytest plugin ([#1146](https://github.com/anam-org/metaxy/pull/1146) by [@danielgafni](https://github.com/danielgafni))
- move metaxy._utils to metaxy.utils ([#1144](https://github.com/anam-org/metaxy/pull/1144) by [@danielgafni](https://github.com/danielgafni))
- adopt Tach to enforce API boundaries ([#1128](https://github.com/anam-org/metaxy/pull/1128) by [@danielgafni](https://github.com/danielgafni))
- rename and move _collect_to_arrow to `metaxy._utils.collect_to_arrow` ([#1130](https://github.com/anam-org/metaxy/pull/1130) by [@danielgafni](https://github.com/danielgafni))

#### Ray

- ensure `Map` type works with `day.data.Dataset.map_batches` ([#1148](https://github.com/anam-org/metaxy/pull/1148) by [@danielgafni](https://github.com/danielgafni))


## v0.1.7 (28-03-2026)

This release adds experimental support for the `Map` datatype and a new Apache Iceberg metadata store.


### :sparkles: Features

- A new experimental `enable_map_datatype` global setting has been added to Metaxy's configuration. When enabled, it uses [`polars-map`](https://pypi.org/project/polars-map/) (must be installed) to preserve the Arrow `Map` type on Polars frames across Metaxy operations. ([#1104](https://github.com/anam-org/metaxy/pull/1104) by [@danielgafni](https://github.com/danielgafni))

#### Claude

- update Claude plugin with `Map` support info ([#1117](https://github.com/anam-org/metaxy/pull/1117) by [@danielgafni](https://github.com/danielgafni))
- update Claude plugin with External Features docs ([#1086](https://github.com/anam-org/metaxy/pull/1086) by [@danielgafni](https://github.com/danielgafni))

#### Clickhouse

- `Map` type support (experimental) ([#1107](https://github.com/anam-org/metaxy/pull/1107) by [@danielgafni](https://github.com/danielgafni))

#### Duckdb

- `Map` type support (experimental) ([#1105](https://github.com/anam-org/metaxy/pull/1105) by [@danielgafni](https://github.com/danielgafni))

#### Deltalake

- `Map` type support (experimental) ([#1108](https://github.com/anam-org/metaxy/pull/1108) by [@danielgafni](https://github.com/danielgafni))

#### Iceberg

- `Map` type support (experimental) ([#1108](https://github.com/anam-org/metaxy/pull/1108) by [@danielgafni](https://github.com/danielgafni))
- add Apache Iceberg metadata store ([#1033](https://github.com/anam-org/metaxy/pull/1033) by [@ashutosh1807](https://github.com/ashutosh1807))


### :bug: Bug Fixes

- do not validate incomplete MetaxyConfig when `extend` is set ([#1087](https://github.com/anam-org/metaxy/pull/1087) by [@danielgafni](https://github.com/danielgafni))

#### Dagster

- log runtime metadata for `Map` type correctly ([#1120](https://github.com/anam-org/metaxy/pull/1120) by [@danielgafni](https://github.com/danielgafni))

#### Iceberg

- use appropriate warehouse path on Windows ([#1094](https://github.com/anam-org/metaxy/pull/1094) by [@danielgafni](https://github.com/danielgafni))


### :book: Docs

- add scope subgroups to changelog ([#1123](https://github.com/anam-org/metaxy/pull/1123) by [@danielgafni](https://github.com/danielgafni))
- update `enable_map_datatype` docstring ([#1121](https://github.com/anam-org/metaxy/pull/1121) by [@danielgafni](https://github.com/danielgafni))
- document `Map` datatype usage and `enable_map_datatype` config option ([#1115](https://github.com/anam-org/metaxy/pull/1115) by [@danielgafni](https://github.com/danielgafni))
- fix broken metaxy.lock glossary link ([#1078](https://github.com/anam-org/metaxy/pull/1078) by [@danielgafni](https://github.com/danielgafni))

#### Iceberg

- refine documentation ([#1095](https://github.com/anam-org/metaxy/pull/1095) by [@danielgafni](https://github.com/danielgafni))


### :hammer_and_wrench: Other Improvements

- release 0.1.7 ([#1124](https://github.com/anam-org/metaxy/pull/1124) by [@danielgafni](https://github.com/danielgafni))
- fix full test suite for windows and macos ([#1125](https://github.com/anam-org/metaxy/pull/1125) by [@danielgafni](https://github.com/danielgafni))
- run all integration tests on core changes ([#1122](https://github.com/anam-org/metaxy/pull/1122) by [@danielgafni](https://github.com/danielgafni))
- add test case for fallback stores with `Map` enabled ([#1119](https://github.com/anam-org/metaxy/pull/1119) by [@danielgafni](https://github.com/danielgafni))
- avoid cancelling in-progress tests in `main` ([#1118](https://github.com/anam-org/metaxy/pull/1118) by [@danielgafni](https://github.com/danielgafni))
- add hypothesis tests for `Map` roundrtip ([#1113](https://github.com/anam-org/metaxy/pull/1113) by [@danielgafni](https://github.com/danielgafni))
- conditionally skip tests for integrations ([#1114](https://github.com/anam-org/metaxy/pull/1114) by [@danielgafni](https://github.com/danielgafni))
- avoid map->struct->map conversion with `enable_map_datatype` ([#1110](https://github.com/anam-org/metaxy/pull/1110) by [@danielgafni](https://github.com/danielgafni))
- fix coverage setup post CI split ([#1101](https://github.com/anam-org/metaxy/pull/1101) by [@danielgafni](https://github.com/danielgafni))
- generate integration matrix with nox ([#1099](https://github.com/anam-org/metaxy/pull/1099) by [@danielgafni](https://github.com/danielgafni))
- split tests for integrations into separate jobs ([#1098](https://github.com/anam-org/metaxy/pull/1098) by [@danielgafni](https://github.com/danielgafni))
- move tests for integrations into tests/ext ([#1097](https://github.com/anam-org/metaxy/pull/1097) by [@danielgafni](https://github.com/danielgafni))
- consolidate MetadataStore instantiation to fixtures ([#1096](https://github.com/anam-org/metaxy/pull/1096) by [@danielgafni](https://github.com/danielgafni))
- split shared test suite for MetadataStore implementations to more granular modules ([#1091](https://github.com/anam-org/metaxy/pull/1091) by [@danielgafni](https://github.com/danielgafni))
- upgrade ty to 0.0.25 ([#1092](https://github.com/anam-org/metaxy/pull/1092) by [@danielgafni](https://github.com/danielgafni))
- add a section for other improvements to changelog ([#1093](https://github.com/anam-org/metaxy/pull/1093) by [@danielgafni](https://github.com/danielgafni))
- delete migrations system ([#1089](https://github.com/anam-org/metaxy/pull/1089) by [@danielgafni](https://github.com/danielgafni))
- split metaxy.config to submodules ([#1088](https://github.com/anam-org/metaxy/pull/1088) by [@danielgafni](https://github.com/danielgafni))

#### Ducklake

- add `Map` datatype tests for DuckLake ([#1116](https://github.com/anam-org/metaxy/pull/1116) by [@danielgafni](https://github.com/danielgafni))


### :heart: New Contributors

- [@ashutosh1807](https://github.com/ashutosh1807) made their first contribution in [#1033](https://github.com/anam-org/metaxy/pull/1033)

## v0.1.6 (18-03-2026)

### :bug: Bug Fixes

#### Sqlalchemy

- the `SQLAlchemy` integration configuration (`[ext.sqlalchemy]` in `metaxy.toml`) has been removed. Please use the code interface instead. ([#1075](https://github.com/anam-org/metaxy/pull/1075) by [@danielgafni](https://github.com/danielgafni))

#### Sqlmodel

- the `SQLModel` integration configuration (`[ext.sqlmodel]` in `metaxy.toml`) has been removed. Please use the code interface instead. ([#1074](https://github.com/anam-org/metaxy/pull/1074) by [@danielgafni](https://github.com/danielgafni))


### :book: Docs

- only display field name in configuration docs ([#1070](https://github.com/anam-org/metaxy/pull/1070) by [@danielgafni](https://github.com/danielgafni))


### :hammer_and_wrench: Other Improvements

- release 0.1.6 ([#1077](https://github.com/anam-org/metaxy/pull/1077) by [@danielgafni](https://github.com/danielgafni))
- update changelog to include change scope ([#1076](https://github.com/anam-org/metaxy/pull/1076) by [@danielgafni](https://github.com/danielgafni))
- move PostgreSQL metadata store to metaxy.ext ([#1069](https://github.com/anam-org/metaxy/pull/1069) by [@danielgafni](https://github.com/danielgafni))


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


### :hammer_and_wrench: Other Improvements

- release 0.1.5 ([#1068](https://github.com/anam-org/metaxy/pull/1068) by [@danielgafni](https://github.com/danielgafni))
- add tag message to changelog template ([#1067](https://github.com/anam-org/metaxy/pull/1067) by [@danielgafni](https://github.com/danielgafni))
- tweak cliff.toml ([#1065](https://github.com/anam-org/metaxy/pull/1065) by [@danielgafni](https://github.com/danielgafni))
- rename extends to extend ([#1064](https://github.com/anam-org/metaxy/pull/1064) by [@danielgafni](https://github.com/danielgafni))
- extend array fields when merging configs ([#1063](https://github.com/anam-org/metaxy/pull/1063) by [@danielgafni](https://github.com/danielgafni))
- adopt deadcode ([#1056](https://github.com/anam-org/metaxy/pull/1056) by [@geoHeil](https://github.com/geoHeil))
- upgraded GHA actions ([#1057](https://github.com/anam-org/metaxy/pull/1057) by [@geoHeil](https://github.com/geoHeil))
- improve git-cliff setup ([#1052](https://github.com/anam-org/metaxy/pull/1052) by [@danielgafni](https://github.com/danielgafni))
- upgrade ty to 0.0.22 ([#1053](https://github.com/anam-org/metaxy/pull/1053) by [@danielgafni](https://github.com/danielgafni))
- add AI policy ([#1051](https://github.com/anam-org/metaxy/pull/1051) by [@danielgafni](https://github.com/danielgafni))
- adopt `git-cliff` ([#1049](https://github.com/anam-org/metaxy/pull/1049) by [@danielgafni](https://github.com/danielgafni))
- adopt Conventional Commits ([#1050](https://github.com/anam-org/metaxy/pull/1050) by [@danielgafni](https://github.com/danielgafni))


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
