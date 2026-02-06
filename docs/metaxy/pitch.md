---
title: "Pitch"
description: "What Metaxy has to offer."
---

# The Metaxy Pitch

Here is why we think Metaxy is so cool. Metaxy is...

## 🧩 Composable

Bring your own... really everything. Metaxy is a universal glue for metadata. Use it with:

<div class="annotate" markdown>

- Your database or storage format of choice to keep metadata where you want. [DuckDB](../integrations/metadata-stores/databases/duckdb.md), [ClickHouse](../integrations/metadata-stores/databases/clickhouse.md) and **20+ databases** via [Ibis](https://ibis-project.org/) (1), **lakehouse storage** formats such as DeltaLake or DuckLake, and other solutions such as [LanceDB](../integrations/metadata-stores/databases/lancedb.md). All of this is available through a unified [interface](../guide/concepts/metadata-stores.md).
- Your favorite dataframe library: Polars, Pandas, or even run **all Metaxy computations in the DB** thanks to [Narwhals](https://narwhals-dev.github.io/narwhals/)
- Orchestrators: see the excellent [Dagster integration](../integrations/orchestration/dagster/index.md) :octopus:
- Compute frameworks like [Ray](../integrations/compute/ray.md). We totally don't care how is **data** (2) produced or where is it stored.
- Version tracking methods. By default, Metaxy uses [Merkle Trees](https://en.wikipedia.org/wiki/Merkle_tree) to track changes in metadata. However, users can provide their own data versions and use content-based hashing techniques (3) if needed.

</div>

1. while we don't (yet) ship native support for all these databases, the base [`IbisMetadataStore`][metaxy.metadata_store.ibis.IbisMetadataStore] can be easily extended to handle additional databases
2. so not the tables but the actual stuff: images, videos, texts, etc.
3. from naive `sha256` to more sophisticated [semantic hashing](https://github.com/MinishLab/semhash)

## 🪨 Reliable

Metaxy is obsessively tested across all supported tabular compute engines. We guarantee to produce [versioning hashes](../guide/concepts/data-versioning.md) that are **consistent across DBs and local** compute engines. We really have tested this very well! (1)
{ .annotate }

1. We have a rich test suite that's run against Linux, Windows and MacOS on all supported Python versions

Metadata is organized in **append-only tables**. Metaxy never attempts to modify historical metadata, (1) ensuring that data integrity is maintained and historical metadata can be easily retrieved.
{ .annotate }

1. But provides the ability to perform hard and soft deletions

## 📈 Scalable

Metaxy is built with **performance** in mind: all operations default to **run in the DB**, the storage layout is designed with the goal of supporting parallel writers and bulk insertions.

Feature definitions can be split across independent Python modules and packages and automatically loaded via packaging entry points. This enables collaboration across teams and projects.

We also have a [Ray integration](../integrations/compute/ray.md) which simplifies working with Metaxy from distributed workflows.

## 🧑‍💻 Developer Friendly

Metaxy provides a clean, intuitive Python API [with syntactic sugar](../guide/concepts/syntactic-sugar.md) that simplifies common feature definitions. The [feature discovery](../guide/concepts/feature-discovery.md) system enables effortless feature dependency management.

The library includes comprehensive **type hints** (1), and utilizes Pydantic for feature definitions. There's first-class support for **local development** (2), **testing**, **preview environments**, and **CI/CD** workflows.
{ .annotate }

1. with all the typing shenanigans you would expect from a project as serious as ours
2. the reference local versioning engine is implemented in [Polars](https://docs.pola.rs/) and [`polars-hash`](https://github.com/ion-elgreco/polars-hash)

The included [CLI](../reference/cli.md) tool allows easy interaction, inspection and visualization of feature graphs, enriched with real metadata and stats. You can even drop your database in one command! (1)
{ .annotate }

1. that's _almost_ a joke

Hopefully this was impressive enough and has sparked some interest in Metaxy!

## What's Next?

- Learn about Metaxy [design choices](./design.md)
- Itching to write some Metaxy code? Jump to [Quickstart](/guide/quickstart/hello.md).
--8<-- "whats-next.md"
