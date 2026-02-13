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
  <a href="https://codecov.io/gh/anam-org/metaxy"><img src="https://codecov.io/gh/anam-org/metaxy/graph/badge.svg" alt="codecov"></a>
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://docs.astral.sh/ty/"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json" alt="Ty"></a>
  <a href="https://prek.j178.dev"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json" alt="prek"></a>
</p>

---

## Metaxy

Metaxy is a pluggable metadata layer for building multimodal Data and ML pipelines. Metaxy manages and tracks **metadata** across complex computational graphs, implements sample and sub-sample versioning, allowing the codebase to evolve over time without friction.

### The problem: sample-level versioning

!!! info annotate

    Data, ML and AI workloads processing **large amounts** of images, videos, audios, or texts (1) can be very expensive to run.
    In contrast to traditional data engineering, re-running the whole pipeline on changes is no longer an option.
    Therefore, it becomes crucially important to correctly implement **incremental processing**, **sample-level versioning** and **prunable updates**.

1. or really any kind of data

These workloads **evolve all the time**, with new data being shipped, bugfixes or algorithm changes introduced, and new features added to the pipeline. This means the pipeline has to be **re-computed frequently**, but at the same time it's important to avoid unnecessary recomputations for individual data samples.

!!! info

    Unnecessary recomputations can waste dozens of thousands of dollars on compute, and battling sample-level orchestration complexity can cost even more in engineering efforts.

--8<-- "data-vs-metadata.md"

In contrast to what one might expect, spinning up a thousand compute nodes is a much easier task with established solutions, while sample-level versioning remains a challenging problem (1).
{ .annotate }

1. it is hard to overestimate the amount of pain [@danielgafni](https://github.com/danielgafni) has endured before building Metaxy

### The solution

Until recently, a general solution for this problem did not exist, but not anymore :tada: !

Metaxy allows creating and updating *feature definitions* which can independently version different *fields* of the same data sample and express granular field-level lineage.

!!! success annotate  "Just Use Metaxy"

    Metaxy has quite a few superpowers:

    - Cache every single sample in the data pipeline. Millions of cache keys can be calculated in under a second (1). Benefit from prunable partial updates.
    - Freedom from storage lock-in. Swap [storage backends](./integrations/metadata-stores/index.md) in development and production environments without breaking a sweat (2).
    - Metaxy is **pluggable**, **declarative**, **composable** and **extensible** (3): use it to build custom integrations and workflows, benefit from emergent capabilities that enable tooling, visualizations and optimizations you didn't even plan for.


1. Our experience at [Anam](https://anam.ai/) with [ClickHouse](./integrations/metadata-stores/databases/clickhouse.md)
2. For example, develop against [DeltaLake](./integrations/metadata-stores/storage/delta.md) and scale production with [ClickHouse](./integrations/metadata-stores/databases/clickhouse.md) without code changes.
3. See our official integrations [here](./integrations/index.md)

And now the killer feature:

!!! tip annotate "Super Granular Data Versioning"

    The feature that makes Metaxy really stand out is the ability to identify **prunable partial data updates** (1) and **skip unnecessary downstream computations**. At the moment of writing, Metaxy is the only available tool that tackles these problems.

1.  which are **very common** in multimodal pipelines

Read [The Pitch](./metaxy/pitch.md) to be impressed even more.

All of this is possible thanks to (1) [Narwhals](https://narwhals-dev.github.io/narwhals/), [Ibis](https://ibis-project.org/), and a few clever tricks.
{ .annotate }

1. we really do stand on the shoulders of giants

## Reliability

Metaxy was [designed](./metaxy/design.md) to handle large amounts of **big metadata** in distributed environments, makes very few assumptions about usage patterns and is non-invasive to the rest of the data pipeline.

Metaxy is fanatically tested across all supported metadata stores, Python versions and platforms [^1]. We guarantee versioning consistency across the supported metadata stores.

We have been dogfooding Metaxy at [Anam](https://anam.ai/) since December 2025. We are running it in production with [ClickHouse](./integrations/metadata-stores/databases/clickhouse.md), [Dagster](./integrations/orchestration/dagster/index.md), and [Ray](./integrations/compute/ray.md) (1), and it's powering all our pipelines that prepare training data for our video generation models.
{ .annotate }

1. and integrations with these tools are probably the most complete at the moment

That being said, Metaxy is still an early project, so while the core functionality is rock solid, some rough edges with other parts of Metaxy are expected.

## Installation

Install Metaxy from [PyPI](https://pypi.org/project/metaxy/):

```shell
uv add metaxy
```

## Quickstart

!!! tip

    Itching to get your hands dirty?
    Head to [Quickstart](./guide/quickstart/quickstart.md).

## What's Next?

Here are a few more useful links:

- Read the [Metaxy Pitch](./metaxy/pitch.md)
- Learn about Metaxy [design choices](./metaxy/design.md)
--8<-- "whats-next.md"

[^1]: The CLI is not tested on Windows yet.
