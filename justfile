ruff:
    uv run ruff check --fix
    uv run ruff format

typecheck:
    uv run basedpyright --level error

sync:
    uv sync --all-extras --all-groups

# Resolve GitHub issue with Claude in an independent git worktree
claude-resolve number prompt="":
    #!/usr/bin/env bash
    set -euxo pipefail
    git worktree add ../worktrees/metaxy-gh-{{number}} origin/main
    cd ../worktrees/metaxy-gh-{{number}}
    direnv allow
    eval "$(direnv export bash)"
    claude --dangerously-skip-permissions "Resolve GitHub issue #{{number}} (do not create commits). {{prompt}}"

docs-build:
    uv run --group docs mkdocs build --clean --strict

docs-serve:
    uv run --group docs mkdocs serve

docs-publish version:
    uv run --group docs --all-extras mike deploy --push --update-aliases {{version}}
