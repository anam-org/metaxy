---
title: "Metaxy Overview"
description: "Introduction to Metaxy."
---

<p align="center">
  <img src="assets/metaxy.svg" alt="Metaxy Logo" width="100"/>
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

---

!!! warning
    Metaxy hasn't been publicly released yet, but you can try the latest dev release:

    ```shell
    pip install --pre metaxy
    ```

---

Metaxy is a composable metadata layer for multi-modal Data and ML pipelines that manages and tracks **metadata**: sample [versions](guide/learn/data-versioning.md), dependencies, and data lineage across complex computational graphs. Metaxy gives you quite a few superpowers:

<div class="annotate" markdown>
- Cache every single sample in the data pipeline. This evaluation runs for millions of samples in under a second. (1)
- Swap [storage backends](./guide/learn/metadata-stores.md) in dev, testing and production environments without changing a single line of code.
- Use the [`mx` CLI](./reference/cli.md) to observe and manage metadata without leaving your terminal
- Metaxy is **composable** and **extensible** (2): use it to build custom integrations and workflows!
</div>

1. Our experience at [Anam](https://anam.ai/) with [ClickHouse](./integrations/metadata-stores/databases/clickhouse/index.md)
2. See our official integrations [here](./integrations/index.md)

Metaxy manages **metadata** while **data** typically (2) lives elsewhere:
{ .annotate }

2. Unless you are a [LanceDB](https://lancedb.com/) fan, in which case [we got you covered](./integrations/metadata-stores/databases/lancedb/index.md)

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

| **Subject** | **Description** |
|---------|-------------|
| **Data** | The actual multi-modal data itself, such as images, audio files, video files, text documents, and other raw content that your pipelines process and transform. |
| **Metadata** | Information about the data, typically including references to where data is stored (e.g., object store keys) plus additional descriptive entries such as video length, file size, format, version, and other attributes. |

The feature that makes Metaxy stand out is the ability to track **partial data dependencies** that are so common in multi-modal pipelines and skip downstream updates when they are not needed.

Metaxy is [agnostic](#about-metaxy) to orchestration frameworks, compute engines, data or [metadata storage](guide/learn/metadata-stores.md). Metaxy has no strict infrastructure requirements, and can scale to handle large amounts of **big metadata**. Metaxy is fanatically tested across all supported Python versions and platforms [^1].

All of this is possible thanks to (1) [Narwhals](https://narwhals-dev.github.io/narwhals/), [Ibis](https://ibis-project.org/), and a few clever tricks.
{ .annotate }

1. we really do stand on the shoulders of giants

## What problem exactly does Metaxy solve?

Data, ML and AI workloads processing **large amounts** of images, videos, audios, or texts (1) can be very expensive to run.
In contrast to traditional data engineering, re-running the whole pipeline on changes is no longer an option.
Therefore, it becomes crucially important to correctly implement incremental processing and sample-level versioning.
{ .annotate }

1. or really any kind of data

These workloads typically aren't stale: they evolve all the time, with new data being shipped, bugfixes or algorithm changes introduced, and new features added to the pipeline. This means the pipeline has to be re-computed frequently, but at the same time it's important to avoid unnecessary recomputations for individual samples in the datasets.

Here are some of the cases where re-computing would be **undesirable**:

- merging two consecutive steps into one (refactoring the graph topology)

- **partial data updates**, e.g. changing only the audio track inside a video file

- backfilling metadata from another source

Correctly distinguishing these scenarios from cases where the feature **should** be re-computed is surprisingly challenging.
Tracking and propagating these changes correctly to the right subset of samples and features can become incredibly complicated and time-consuming.
Until now, a general solution for this problem did not exist, but this is not the case anymore.

## Metaxy's solution

<div class="annotate" markdown>

1. Metaxy builds a *versioned graph* from declarative [feature definitions](./guide/learn/feature-definitions.md) and tracks [version changes](./guide/learn/data-versioning.md) across individual samples.

2. Metaxy introduces a unique [field-level](./guide/learn/feature-definitions.md#field-level-dependencies) dependency system to express partial data dependencies and avoiding unnecessary downstream recomputations. Each sample holds a dictionary mapping its fields to their respective versions. (1)

3. Metaxy implements a general [`MetadataStore`](./guide/learn/metadata-stores.md) interface that enables it to work with any kind of storage system, be it an analytical database or a lakehouse format.

</div>

1. for example, a `video` sample could independently version the frames and the audio track: `{"audio": "asddsa", "frames": "joasdb"}`

## Quickstart

Head to [Quickstart](./guide/overview/quickstart.md) (WIP!).

## About Metaxy

Metaxy is:

<div class="annotate" markdown>

- **🧩 composable** --- bring your own everything!

    - supports [DuckDB](./integrations/metadata-stores/databases/duckdb/index.md), [ClickHouse](./integrations/metadata-stores/databases/clickhouse/index.md), and **20+ databases** via [Ibis](https://ibis-project.org/), **lakehouse storage** formats such as DeltaLake or DuckLake, and other solutions such as [LanceDB](./integrations/metadata-stores/databases/lancedb/index.md)

    - is **agnostic to tabular compute engines**: Polars, Spark, Pandas, and databases thanks to [Narwhals](https://narwhals-dev.github.io/narwhals/)

    - we totally don't care how is the multi-modal **data** produced or where is it stored: Metaxy is responsible for yielding input metadata and writing output metadata

    - blends right in with orchestrators: see the excellent [Dagster integration](./integrations/orchestration/dagster/index.md) :octopus:

- **🪨 rock solid** when it matters:

    - [versioning](./guide/learn/data-versioning.md) is guaranteed to be **consistent across DBs and local** (1) compute engines. We really have tested this very well!

    - unique [field-level dependency system](./guide/learn/feature-definitions.md#field-level-dependencies) prevents unnecessary recomputations for features that depend on partial data

    - metadata is **append-only** to ensure data integrity and immutability, but Metaxy provides the tooling to perform cleanup on-demand

- **📈 scalable**:

    - is built with **performance** in mind: all operations default to **run in the DB**, parallel writers are supported natively

    - [Ray integration](./integrations/compute/ray.md) enables distributed compute for large-scale workloads

    - supports **feature organization and discovery** patterns such as packaging entry points. This enables collaboration across teams and projects.

- **🧑‍💻 dev friendly**:

    - clean, intuitive Python API [with syntactic sugar](./guide/learn/syntactic-sugar.md) that simplifies common feature definitions

    - [feature discovery](./guide/learn/feature-discovery.md) system for effortless dependency management

    - comprehensive **type hints** with all the typing shenanigans like `@overload`. Metaxy utilizes Pydantic for feature definitions.

    - first-class support for **local development**, **testing**, **preview environments**, and **CI/CD** workflows

    - [CLI](./reference/cli.md) tool for easy interaction, inspection and visualization of feature graphs, enriched with real metadata and stats

    - [integrations](integrations/index.md) with popular tools such as Dagster, DuckDB, SQLModel and others

</div>

1. the local versioning engine is implemented in [Polars](https://docs.pola.rs/) and [`polars-hash`](https://github.com/ion-elgreco/polars-hash)

## What's Next?

- Itching to write some Metaxy code? Continue to [Quickstart](./guide/overview/quickstart.md) (WIP!).

- Learn more about [feature definitions](./guide/learn/feature-definitions.md) or [versioning](./guide/learn/data-versioning.md)

- View complete, end-to-end [examples](./examples/index.md)

- Explore [Metaxy integrations](integrations/index.md)

- Use Metaxy [from the command line](./reference/cli.md)

- Learn how to [configure Metaxy](./reference/configuration.md)

- Get lost in our [API Reference](./reference/api/index.md)

[^1]: The CLI is not tested on Windows yet.
