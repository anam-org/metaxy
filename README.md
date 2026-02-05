<p align="center">
  <img src="https://raw.githubusercontent.com/anam-org/metaxy/main/docs/assets/metaxy.svg" alt="Metaxy Logo" width="100"/>
</p>

<h1 align="center">Metaxy</h1>

<p align="center">
  <a href="https://pypi.org/project/metaxy/"><img src="https://img.shields.io/pypi/v/metaxy.svg?color=4644ad" alt="PyPI version"></a>
  <a href="https://pypi.org/project/metaxy/"><img src="https://img.shields.io/pypi/pyversions/metaxy.svg?color=4644ad" alt="Python versions"></a>
  <a href="https://pypi.org/project/metaxy/"><img src="https://img.shields.io/pypi/dm/metaxy.svg?color=4644ad" alt="PyPI downloads"></a>
  <a href="https://github.com/anam-org/metaxy/actions/workflows/main.yml"><img src="https://github.com/anam-org/metaxy/actions/workflows/main.yml/badge.svg" alt="CI"></a>
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://docs.astral.sh/ty/"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json" alt="Ty"></a>
  <a href="https://prek.j178.dev"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json" alt="prek"></a>
</p>

Metaxy is a metadata layer for **multi-modal Data and ML pipelines**. Metaxy tracks lineage and versioning across complex computational graphs for multi-modal datasets. Metaxy can cache every single sample and scale to handle millions of them.

Metaxy manages **metadata** while **data** typically lives elsewhere:

```
┌─────────────────────────────────┐          ┌─────────────────────────┐
│      Metadata (Metaxy)          │          │   Data (e.g., S3)       │
├──────┬──────────┬──────┬────────┤          │                         │
│  ID  │   path   │ size │version │          │  📦 s3://my-bucket/     │
├──────┼──────────┼──────┼────────┤          │                         │
│ img1 │ s3://... │ 2.1M │a3fdsf  │ ────────>│    ├─ img1.jpg          │
│ img2 │ s3://... │ 1.8M │b7e123  │ ────────>│    ├─ img2.jpg          │
└──────┴──────────┴──────┴────────┘          └─────────────────────────┘
```

The feature that makes Metaxy stand out is the ability to track **field-level dependencies** and detect **prunable updates** — updates that don't trigger change propagation through certain paths in the dependency graph because they modify fields that aren't dependencies of those downstream features. For example, updating audio upstream of a face recognition step allows pruning the face recognition branch since it only depends on video frames. This problem is specific to multi-modal pipelines and doesn't typically emerge in traditional data engineering.

Metaxy's goal is to provide a standard instrument for any kind of multi-modal (or just purely tabular) **incremental** pipelines, standardizing dependency specification, versioning, partial data dependencies, and manipulations over metadata. Or, in short, to be a universal glue for incremental data pipelines.

Metaxy is very reliable and is fanatically tested across all supported Python versions and platforms [^1].

## Documentation

Read the [docs](https://docs.metaxy.io) to learn more.

## Installation

> [!WARNING]
> Metaxy hasn't been publicly released yet, but you can try the `main` branch:
> ```shell
> pip install git+https://github.com/metaxy-dev/metaxy.git
> ```

## Using Metaxy

Metaxy is highly pluggable and generally can be used with any kind of incremental pipelines, storage, metadata storage, and dataframe libraries.

Metaxy provides integrations with popular tools such as [Dagster](https://docs.metaxy.io/main/integrations/orchestration/dagster), [Ray](https://docs.metaxy.io/main/integrations/compute/ray), [ClickHouse](https://docs.metaxy.io/main/integrations/metadata-stores/databases/clickhouse), [DeltaLake](https://docs.metaxy.io/main/integrations/metadata-stores/storage/delta/), [SQLModel](https://docs.metaxy.io/main/integrations/plugins/sqlmodel/).

The full list can be found [here](https://docs.metaxy.io/main/integrations).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

<!-- scarf.sh for telemetry collection ( it does not record personal data such as IP addresses) -->
<img referrerpolicy="no-referrer" src="https://telemetry.metaxy.io/a.png?x-pxid=22cb75dc-201e-4a72-9fb2-c3a53ce9207e&page=README.md" />

[^1]: The CLI is not tested on Windows yet.
