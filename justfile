ruff:
    uv run ruff check --fix
    uv run ruff format

generate-docs-cli:
    uv run cyclopts generate-docs src/metaxy/cli/app.py:app -f md -o docs/cli.md

sync:
    uv sync --all-extras --all-groups
