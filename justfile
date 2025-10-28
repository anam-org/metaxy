ruff:
    uv run ruff check --fix
    uv run ruff format

typecheck:
    uv run basedpyright --level error

generate-docs-cli:
    uv run cyclopts generate-docs src/metaxy/cli/app.py:app -f md -o docs/cli.md

sync:
    uv sync --all-extras --all-groups

# Resolve GitHub issue with Claude in an independent git worktree
claude-resolve number:
    git worktree add resolve-{{number}}
    cd resolve-{{number}}
    gt checkout main
    gt get
    direnv allow
    source ./.venv/bin/activate
    ISSUE=gh issue view {{number}}
    claude -p "Please solve GitHub issue (do not create commits) #{{number}}:\n\n$ISSUE"
