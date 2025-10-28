ruff:
    uv run ruff check --fix
    uv run ruff format

typecheck:
    uv run basedpyright --level error

sync:
    uv sync --all-extras --all-groups

# Resolve GitHub issue with Claude in an independent git worktree
claude-resolve number:
    git worktree add ../metaxy-worktrees/resolve-{{number}} origin/main
    cd ../metaxy-worktrees/resolve-{{number}}
    direnv allow
    claude "Resolve GitHub issue (do not create commits) #{{number}}"

docs-build:
    uv run --group docs mkdocs build --clean --strict

docs-serve:
    uv run --group docs mkdocs serve

docs-publish version:
    uv run --group docs --all-extras mike deploy --push --update-aliases {{version}}
