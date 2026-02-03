---
title: "Introduction"
description: "A high level introduction to Metaxy."
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

## Metaxy

<div class="annotate" markdown>

Metaxy is a pluggable metadata layer for building multi-modal Data and ML pipelines that manages and tracks **metadata** across complex computational graphs, including row-level [versions](guide/learn/data-versioning.md) (1), while allowing the codebase to evolve over time without friction. Metaxy gives you quite a few superpowers:

- Cache every single sample in the data pipeline. Millions of cache keys can be calculated in under a second. (2)
- Freedom from storage lock-in. Swap [storage backends](./integrations/metadata-stores/index.md) in development and production environments without breaking a sweat (3).
- Use the [`mx` CLI](./reference/cli.md) to observe and manage metadata without leaving the comfort of your terminal.
- Metaxy is **composable** and **extensible** (4): use it to build custom integrations and workflows!

</div>

1. And even more granular [partial data versions](http://localhost:8000/guide/learn/data-versioning/#samples)
2. Our experience at [Anam](https://anam.ai/) with [ClickHouse](./integrations/metadata-stores/databases/clickhouse.md)
3. For example, develop against [DeltaLake](./integrations/metadata-stores/storage/delta.md) and scale production with [ClickHouse](./integrations/metadata-stores/databases/clickhouse.md) without code changes.
4. See our official integrations [here](./integrations/index.md)


!!! tip annotate "Granular Data Versioning"

    The feature that makes Metaxy really stand out is the ability to track **partial data dependencies** (1) and **skip downstream updates** unless the exactly required subset of upstream data has changed. At the moment of writing, Metaxy is the only available tool that tackles these problems.

1.  which are **very common** in multi-modal pipelines, for example when you only need to process video frames and not the audio tracks


All of this is possible thanks to (1) [Narwhals](https://narwhals-dev.github.io/narwhals/), [Ibis](https://ibis-project.org/), and a few clever tricks.
{ .annotate }

1. we really do stand on the shoulders of giants


??? abstract annotate "Data vs Metadata Clarifications"

    Metaxy manages **metadata** while **data** typically (1) lives elsewhere:
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

1.  Unless you are a [LanceDB](https://lancedb.com/) fan, in which case [we got you covered](./integrations/metadata-stores/databases/lancedb.md)

## Reliability

Metaxy is fanatically tested across all supported metadata stores, Python versions and platforms [^1]. We guarantee versioning consistency across the supported metadata stores.

Metaxy was built to handle large amounts of **big metadata** in distributed environments, makes very little assumptions about usage patterns and cannot enter an inconsistent state.

We have been dogfooding Metaxy since December 2025 at Anam. We are running it in production with [ClickHouse](./integrations/metadata-stores/databases/clickhouse.md), [Dagster](./integrations/orchestration/dagster/index.md), and [Ray](./integrations/compute/ray.md) (1).
{ .annotate }

1. and integrations with these tools are probably the most complete at the moment

## Installation

!!! warning
    Metaxy hasn't been publicly released yet, but you can try the latest dev release:

    ```shell
    pip install --pre metaxy
    ```

## Quickstart

!!! tip

    Urging to get your hands dirty?
    Head to [Quickstart](./guide/overview/quickstart.md) (WIP!).

## What is the problem again?

!!! info annotate

    Data, ML and AI workloads processing **large amounts** of images, videos, audios, or texts (1) can be very expensive to run.
    In contrast to traditional data engineering, re-running the whole pipeline on changes is no longer an option.
    Therefore, it becomes crucially important to correctly implement incremental processing and sample-level versioning.

1. or really any kind of data

These workloads often **aren't stale**: they **evolve all the time**, with new data being shipped, bugfixes or algorithm changes introduced, and new features added to the pipeline. This means the pipeline has to be **re-computed frequently**, but at the same time it's important to avoid unnecessary recomputations for individual data samples.

Here are some of the cases where re-computing would be **undesirable**:

- merging two consecutive steps into one (refactoring the graph topology)
- **partial data updates**, e.g. changing only the audio track inside a video file
- backfilling metadata from another source

Correctly identifying these scenarios while also **re-computing the feature when it should be** is surprisingly challenging, and tracking and propagating these changes correctly to the right subset of samples and features can become incredibly complicated and time-consuming.

## So what can we do about this?

Sounds really bad, right? Yes, and it is (1). Until recently, a general solution for this problem did not exist, but not anymore :tada: !
{ .annotate }

1. I cannot overestimate the amount of hair pulling I've endured before making Metaxy

!!! success  "Just Use Metaxy"

    Metaxy solves this!

<div class="annotate" markdown>

1. Metaxy builds a *versioned graph* from declarative [feature definitions](./guide/learn/feature-definitions.md) and tracks [version changes](./guide/learn/data-versioning.md) across individual samples. These computations can be scaled to run on millions of samples.

2. Metaxy introduces a unique [field-level](./guide/learn/feature-definitions.md#field-level-dependencies) dependency system to express partial data dependencies and avoiding unnecessary downstream recomputations. Each sample holds a dictionary mapping its fields to their respective versions. (1)

3. Metaxy implements a general [`MetadataStore`](./guide/learn/metadata-stores.md) interface that enables users to interact with storage systems -- be it an analytical database or a LakeHouse -- in the same way.

</div>

1. for example, a `video` sample could independently version the frames and the audio track: `{"audio": "asddsa", "frames": "joasdb"}`

## More Information

Here are a few more useful links:

<div class="annotate" markdown>

- Take your time and read a bit more (1) about Metaxy [here](./overview.md)
- Jump to [Quickstart](./guide/overview/quickstart.md) if you just can't wait (WIP!)
- Abstract Metaxy concepts are [discussed here](./guide/learn/index.md)
- View complete, end-to-end [examples](./examples/index.md)
- Explore [Metaxy integrations](integrations/index.md)
- Use Metaxy [from the command line](./reference/cli.md)
- Learn how to [configure Metaxy](./reference/configuration.md)
- Get lost in our [API Reference](./reference/api/index.md)

</div>

1. just one more page, I promise, just one more page

[^1]: The CLI is not tested on Windows yet.
