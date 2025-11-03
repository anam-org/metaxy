# Contributing

We are in early and active development. Contributors are very welcome!

## Development

Setting up the environment:

```shell
uv sync --all-extras
uv run prek install
```

You are also expected to install system dependencies such as `clickhouse` and others. These can be found in `flake.nix`.

### For happy Nix users

`Nix` and `direnv` users can flex with `direnv allow` - this will automatically setup the environment for you, including all system dependencies and Python packages.
We also have Nix dev shells for all supported Python versions:

```shell
nix develop  # enters a shell with the lowest supported Python version
nix develop '.#"python3.11"'  # enters a shell with Python 3.11
```

## Testing

We use `pytest` for testing.

```shell
uv run pytest
```

### Snapshot-testing

Many of the tests snapshot some data that is likely to change (such as data versions) via [syrupy](https://github.com/syrupy-project/syrupy).
To allow snapshots to be updated (this must be an explicit decision), add `--snapshot-update` to Pytest arguments.

### Testing the examples

All examples for this project **must bundle integration tests** placed in `tests/examples`.
