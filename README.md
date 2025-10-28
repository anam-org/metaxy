# Metaxy

# Metaxy 🌌

Metaxy is a feature metadata management system for ML pipelines that tracks feature versions, dependencies, and data lineage across complex computation graphs. Metaxy supports incremental computations, sample-level versioning, field-level versioning, and more.

Read the [docs](https://anam-org.github.io/metaxy) to learn more.

Metaxy is:

- **🧩 composable** --- bring your own everything!

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

Now let's get started with Metaxy! Learn more in [docs](https://anam-org.github.io/metaxy/)

## Development

Setting up the environment:

```shell
uv sync --all-extras
uv run prek install
```

## Examples

See [examples](https://github.com/anam-org/metaxy/tree/main/examples).
