# Contributing

We are in early and active development. Contributors are very welcome!

## Development

We are using [Devenv](https://devenv.sh/) to manage the development environment, except the Python packages, which are managed by `uv`. We also recommend using [Direnv](https://direnv.net/) to automatically build and activate the environment whenever you enter the project directory.s

1. Install `direnv`, `devenv` and `uv`
2. Enter the project directory and run `direnv allow` to setup the environment
3. Run `prek install` (needs to be done only once) to install the pre-commit Git hooks

To reinstall Python dependencies, run `uv sync --all-extras`.

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/). PRs are squash-merged, so the **PR title becomes the commit message** in `main`. This means the PR title must follow conventional commit format — individual commit messages on the branch don't matter. A CI check enforces this.

```
<type>(<optional scope>): <description>
```

| Type | Purpose |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Maintenance (deps, CI, releases) |

Append `!` after the type/scope for breaking changes.

Examples:

```
feat: add BigQuery metadata store support
feat(cli): add --format flag to metaxy inspect
fix: resolve race condition in metadata write
refactor: simplify version calculation logic
fix!: rename MetadataStore.push to MetadataStore.sync
```

Types `chore`, `ci`, `build`, `test`, and `style` are excluded from the changelog.

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

## Building the docs

Run `uv run mkdocs serve` to start the documentation server.
