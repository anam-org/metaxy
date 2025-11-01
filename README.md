# 🌌 Metaxy

Metaxy is a metadata layer for multi-modal Data and ML pipelines that tracks feature versions, dependencies, and data lineage across complex computation graphs.

Metaxy supports incremental computations, sample-level versioning, field-level versioning, and more.

> [!WARNING]
> This project is as raw as a steak still saying ‘moo.’

Read the [few docs we have](https://docs.metaxy.io) to learn more.

<img referrerpolicy="no-referrer-when-downgrade" src="https://telemetry.metaxy.io/a.png?x-pxid=349d49a2-9825-489a-973a-e87096d78a7f&page=README.md" />

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

## Examples

See [examples](https://github.com/anam-org/metaxy/tree/main/examples).
