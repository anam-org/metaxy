<p align="center">
  <img src="https://raw.githubusercontent.com/anam-org/metaxy/main/docs/assets/metaxy.svg" alt="Metaxy Logo" width="100"/>
</p>

<h1 align="center">Metaxy</h1>

<p align="center">
  <a href="https://pypi.org/project/metaxy/"><img src="https://img.shields.io/pypi/v/metaxy.svg?color=4644ad" alt="PyPI version"></a>
  <a href="https://pypi.org/project/metaxy/"><img src="https://img.shields.io/pypi/pyversions/metaxy.svg?color=4644ad" alt="Python versions"></a>
  <a href="https://pypi.org/project/metaxy/"><img src="https://img.shields.io/pypi/dm/metaxy.svg?color=4644ad" alt="PyPI downloads"></a>
  <a href="https://github.com/anam-org/metaxy/actions/workflows/QA.yml"><img src="https://github.com/anam-org/metaxy/actions/workflows/QA.yml/badge.svg" alt="CI"></a>
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://docs.astral.sh/ty/"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json" alt="Ty"></a>
  <a href="https://prek.j178.dev"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json" alt="prek"></a>
</p>

Metaxy is a metadata layer for **multi-modal Data and ML pipelines**. Metaxy tracks feature versions, data dependencies - including multi-modal components - and lineage across complex computational graphs.

Metaxy's goal is to provide a standard instrument for any kind of multi-modal (or just purely tabular) **incremental** pipelines, standardizing dependency specification, versioning, partial data dependencies, and manipulations with metadata.

Read the [docs](https://anam-org.github.io/metaxy) to discover more about Metaxy.

## Installation

**Warning**: Metaxy hasn't been publicly released yet, but you can try the latest dev release:

```shell
pip install --pre metaxy
```

## Integrating Metaxy in your project

Metaxy is highly pluggable and generally can be used with any kind of incremental pipelines, storage, metadata storage, and dataframe libraries.

Metaxy provides integrations with popular tools such as [Dagster](https://anam-org.github.io/metaxy/main/integrations/orchestration/dagster), [ClickHouse](https://anam-org.github.io/metaxy/main/integrations/metadata-stores/databases/clickhouse), [DeltaLake](https://anam-org.github.io/metaxy/main/integrations/metadata-stores/storage/delta/), [SQLModel](https://anam-org.github.io/metaxy/main/integrations/plugins/sqlmodel/).

The full list can be found [here](https://anam-org.github.io/metaxy/main/integrations).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

<img referrerpolicy="no-referrer" src="https://static.scarf.sh/a.png?x-pxid=22cb75dc-201e-4a72-9fb2-c3a53ce9207e&page=README.md" />
