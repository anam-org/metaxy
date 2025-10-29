# 🌌 Metaxy

Metaxy is a feature metadata management system for ML pipelines that tracks feature versions, dependencies, and data lineage across complex computation graphs. Metaxy supports incremental computations, sample-level versioning, field-level versioning, and more.

> [!WARNING]
> This project is as raw as a steak still saying ‘moo.’

Read the [few docs we have](https://anam-org.github.io/metaxy) to learn more.

What's so special about this project? Metaxy is:

- **🧩 composable** -- bring your own everything!

    - supports DuckDB, ClickHouse, and **20+ databases** via [Ibis](https://ibis-project.org/)
    - supports **lakehouse storage** formats such as DeltaLake or DuckLake
    - is **agnostic to tabular compute engines**: Polars, Spark, Pandas, and databases thanks to [Narwhals](https://narwhals-dev.github.io/narwhals/)
    - we totally don't care how is the multi-modal **data** produced or where is it stored: Metaxy is responsible for yielding input metadata and writing output metadata

- **🪨 rock solid** where it matters:

    - [data versioning](https://anam-org.github.io/metaxy/learn/data-versioning.md) is guaranteed to be **consistent across DBs or in-memory** compute engines. We really have tested this very well!
    - changes to topology, feature versioning, or individual samples **ruthlessly propagate downstream**
    - unique **field-level dependency system** prevents unnecessary recomputations for features that depend on partial data
    - metadata is **append-only** to ensure data integrity and immutability. Users can perform cleanup if needed (Metaxy provides tools for this).

- **🤸 flexible** to work around restrictions consciously:

    - has a **migrations system** to compensate for reconciling data versions and metadata when computations are not desired

- **📈 scalable**:

    - supports **feature organization and discovery** patterns such as packaging entry points. This enables collaboration across teams and projects.
    - is built with **performance** in mind: all operations default to **run in the DB**, Metaxy does not stand in the way of metadata flow

- **🧑‍💻 dev friendly**:

    - clean, **intuitive Python API** that stays out of your way when you don't need it
    - [feature discovery](https://anam-org.github.io/metaxy/learn/feature-discovery.md) system for effortless dependency management
    - comprehensive **type hints** and Pydantic integration for excellent IDE support
    - first-class support for **local development, testing, preview environments, CI/CD**
    - [CLI](https://anam-org.github.io/metaxy/reference/cli.md) tool for easy interaction, inspection and visualization of feature graphs, enriched with real metadata and stats
    - integrations with popular tools such as [SQLModel](https://anam-org.github.io/metaxy/learn/integrations/sqlmodel.md), Dagster, and Ray.
    - [testing helpers](https://anam-org.github.io/metaxy/learn/testing.md) that you're going to appreciate

Now let's get started with Metaxy! Learn more in the [docs](https://anam-org.github.io/metaxy/),

## Development

Setting up the environment:

```shell
uv sync --all-extras
uv run prek install
```

You are also expected to install system dependencies such as `clickhouse` and others. These can be found in `flake.nix`.

### For happy Nix users

`Nix` and `direnv` users can flex with `direnv allow` - this will automatically setup the environment for you, including all system dependencies and Python packages. We also have Nix dev shells for all supported Python versions:

```shell
nix develop  # enters a shell with the lowest supported Python version
nix develop '.#"python3.11"'  # enters a shell with Python 3.11
```

Since we are often dealing with external binaries that expect to find their dependencies in normal locations like `/usr/lib`, we also append these locations to `LD_LIBRARY_PATH`. This makes the `devShell` slightly but life much better.

### GitHub Actions

Metaxy is continuously tested on GitHub Actions. You can find the workflow file in `.github/workflows/QA.yml`. This workflow installs system dependencies with Nix and Python packages with `uv`.  The steps use the Nix `devShell` which is **almost** pure.

## Examples

See [examples](https://github.com/anam-org/metaxy/tree/main/examples).
