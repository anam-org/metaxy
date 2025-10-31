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
    git checkout -b gt-{{number}}
    gt track gt-{{number}}
    claude --dangerously-skip-permissions "Resolve GitHub issue #{{number}} (do not create commits). {{prompt}}"


# Create a GitHub issue with Claude and start working on it in an independent git worktree
claude-draft-solution branch prompt:
    #!/usr/bin/env bash
    set -euxo pipefail
    git worktree add ../worktrees/metaxy-{{branch}} origin/main
    cd ../worktrees/metaxy-{{branch}}
    direnv allow
    eval "$(direnv export bash)"
    git checkout -b {{branch}}
    gt track {{branch}}
    claude --dangerously-skip-permissions "{{prompt}}. create a new GitHub issue if this is a new problem -- do not ask for confirmation, I explicitly allow creating it, or search for existing one if the issue number was not mentioned. @agent-planner a plan together with @agent-python-dev. Wait for my confirmation before proceeding with the implementation. Do not create commits."

docs-build:
    uv run --group docs mkdocs build --clean --strict

docs-serve:
    uv run --group docs mkdocs serve

docs-publish version:
    git branch -D gh-pages
    git fetch origin gh-pages
    uv run --group docs --all-extras mike deploy --push --update-aliases {{version}}


slides-serve:
    cd docs/slides && bun dev slides-introduction
