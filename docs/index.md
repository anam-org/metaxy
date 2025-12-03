<p align="center">
  <img src="assets/metaxy.svg" alt="Metaxy Logo" width="100"/>
</p>

<h1 align="center">Metaxy</h1>

<p align="center">
  <a href="https://pypi.org/project/metaxy/"><img src="https://img.shields.io/pypi/v/metaxy.svg?color=4644ad" alt="PyPI version"></a>
  <a href="https://pypi.org/project/metaxy/"><img src="https://img.shields.io/pypi/pyversions/metaxy.svg?color=4644ad" alt="Python versions"></a>
  <a href="https://pypi.org/project/metaxy/"><img src="https://img.shields.io/pypi/dm/metaxy.svg?color=4644ad" alt="PyPI downloads"></a>
  <a href="https://github.com/anam-org/metaxy/actions/workflows/QA.yml"><img src="https://github.com/anam-org/metaxy/actions/workflows/QA.yml/badge.svg" alt="CI"></a>
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/linting-ruff-4644ad" alt="Ruff"></a>
  <a href="https://docs.basedpyright.com"><img src="https://img.shields.io/badge/basedpyright-checked-4644ad" alt="basedpyright - checked"></a>
  <a href="https://prek.j178.dev"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/assets/badge/v0.json" alt="prek"></a>
</p>

---

!!! warning
    Metaxy hasn't been publicly released yet, but you can try the latest dev release:

    ```shell
    pip install --pre metaxy
    ```

---

Metaxy is a metadata layer for multi-modal Data and ML pipelines that manages and tracks **metadata**: sample [versions](learn/data-versioning.md), dependencies, and data lineage across complex computational graphs.

It's [agnostic](#about-metaxy) to orchestration frameworks, compute engines, data and [metadata storage](learn/metadata-stores.md).

It has no strict infrastructure requirements and can run computations in external databases or locally.

It can scale to handle large amounts of **big metadata**.

## What problem exactly does Metaxy solve?

Data, ML and AI workloads processing **large amounts** of images, videos, audios, texts, or any other kind of data can be very expensive to run.
In contrast to traditional data engineering, re-running the whole pipeline on changes is no longer an option.
Therefore, it becomes crucially important to correctly implement incremental processing and sample-level versioning.

Typically, a **feature** has to be re-computed in one of the following scenarios:

- upstream data changes

- bug fixes or algorithmic changes

But correctly distinguishing these scenarios from cases where the feature **should not** be re-computed is surprisingly challenging. Here are some of the cases where it would be undesirable:

- merging two consecutive steps into one (refactoring the graph topology)

- **partial data updates**, e.g. changing only the audio track inside a video file

- backfilling metadata from another source

Tracking and propagating these changes correctly to the right subset of samples and features can become incredibly complicated.
Until now, a general solution for this problem did not exist, but this is not the case anymore.

## Metaxy's solution

Metaxy solves the first set of problems with a **feature** and **field** dependency system, and the second set with a **migrations** system.

Metaxy builds a *versioned graphs* from feature definitions and tracks version changes.

## Quickstart

Head to [Quickstart](./overview/quickstart.md) (WIP!).

## About Metaxy

Metaxy is:

- **🧩 composable** --- bring your own everything!

    - supports DuckDB, ClickHouse, and **20+ databases** via [Ibis](https://ibis-project.org/)
    - supports **lakehouse storage** formats such as DeltaLake or DuckLake
    - is **agnostic to tabular compute engines**: Polars, Spark, Pandas, and databases thanks to [Narwhals](https://narwhals-dev.github.io/narwhals/)
    - we totally don't care how is the multi-modal **data** produced or where is it stored: Metaxy is responsible for yielding input metadata and writing output metadata

- **🤸 flexible** to work around restrictions consciously:

    - [features](./learn/feature-definitions.md) are defined as [Pydantic](https://docs.pydantic.dev/latest/) models, leveraging Pydantic's type safety guarantees, rich validation system, and allowing inheritance patterns to stay DRY
    - has a **migrations system** to compensate for reconciling field provenances and metadata when computations are not desired

- **🪨 rock solid** when it matters:

    - [field provenance](./learn/data-versioning.md) is guaranteed to be **consistent across DBs or in-memory** compute engines. We really have tested this very well!
    - changes to topology, feature versioning, or individual samples **ruthlessly propagate downstream**
    - unique [field-level dependency system](./learn/feature-definitions.md#field-level-dependencies) prevents unnecessary recomputations for features that depend on partial data
    - metadata is **append-only** to ensure data integrity and immutability. Users can perform cleanup if needed (Metaxy provides tools for this).

- **📈 scalable**:

    - supports **feature organization and discovery** patterns such as packaging entry points. This enables collaboration across teams and projects.
    - is built with **performance** in mind: all operations default to **run in the DB**, Metaxy does not stand in the way of metadata flow

- **🧑‍💻 dev friendly**:

    - clean, [intuitive Python API](./learn/syntactic-sugar.md) that stays out of your way when you don't need it
    - [feature discovery](./learn/feature-discovery.md) system for effortless dependency management
    - comprehensive **type hints** and Pydantic integration for excellent IDE support
    - first-class support for **local development, testing, preview environments, CI/CD**
    - [CLI](./reference/cli.md) tool for easy interaction, inspection and visualization of feature graphs, enriched with real metadata and stats
    - [integrations](integrations/index.md) with popular tools such as SQLModel and Dagster.
    - [testing helpers](./learn/testing.md) that you're going to appreciate

## What's Next?

- Itching to write some Metaxy code? Continue to [Quickstart](./overview/quickstart.md) (WIP!).

- Learn more about [feature definitions](./learn/feature-definitions.md) or [versioning](./learn/data-versioning.md)

- Explore [Metaxy integrations](integrations/index.md)

- Use Metaxy [from the command line](./reference/cli.md)

- Learn how to [configure Metaxy](./reference/configuration.md)

- Get lost in our [API Reference](./reference/api/index.md)
