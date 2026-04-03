fmt:
    dprint fmt

ruff:
    uv run ruff check --fix
    uv run ruff format

typecheck:
    uv run ty check

nox *args:
    uv run nox {{args}}

sync:
    uv sync --all-extras --all-groups

new-worktree branch:
    #!/usr/bin/env bash
    set -euxo pipefail
    git worktree add ../worktrees/metaxy-{{branch}} origin/$(git branch --show-current)
    cd ../worktrees/metaxy-{{branch}}
    direnv allow
    eval "$(direnv export bash)"
    git checkout -b {{branch}}
    gt track {{branch}}


claude:
    claude --dangerously-skip-permissions --verbose

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

claude-resolve-conflicts prompt="":
    claude --dangerously-skip-permissions -p "resolve git conflicts. If really in doubt, ask the user for help. {{prompt}}"

docs-build:
    uv run --group docs mkdocs build --clean --strict

docs-build-full:
    METAXY_BUILD_PUBLICATIONS=1 METAXY_SYNC_DOCS_MOUNTS=1 uv run --group docs mkdocs build --clean --strict

docs-serve:
    METAXY_SYNC_DOCS_MOUNTS=1 uv run --group docs mkdocs serve --clean --livereload

docs-publish version:
    git branch -D gh-pages
    git fetch origin gh-pages
    METAXY_SYNC_DOCS_MOUNTS=1 uv run --group docs --all-extras mike deploy --push --update-aliases {{version}}

test-and-submit:
    just ruff
    just typecheck
    uv run pytest --snapshot-update
    dprint fmt
    git add .
    gt modify
    gt sync
    gt ss --no--interactive

init-example name:
    uv init --lib --name {{name}} examples/{{name}}
    uv add --project examples ./examples/{{name}}
    uv add --project ./examples/{{name}} . --editable

# Preview unreleased changelog entries
changelog-preview:
    git cliff --unreleased --strip all

changelog:
    git cliff -o CHANGELOG.md

# Create a release (version auto-detected from commits, or manually specified)
release bump="" message="":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "{{bump}}" ]; then
        uv version --bump {{bump}}
    else
        uv version "$(git cliff --bumped-version | sed 's/^v//')"
    fi
    version="v$(uv version --short)"
    echo "__version__ = \"$(uv version --short)\"" > src/metaxy/_version.py
    if [ -n "{{message}}" ]; then
        git cliff --tag "$version" --with-tag-message "{{message}}" -o CHANGELOG.md
    else
        git cliff --tag "$version" -o CHANGELOG.md
    fi

# Create an annotated tag for the current version, opening the editor for the message
tag message="":
    #!/usr/bin/env bash
    set -euo pipefail
    version="v$(uv version --short)"
    if [ -n "{{message}}" ]; then
        git tag --annotate --cleanup=verbatim --message "{{message}}" --edit "$version"
    else
        git tag --annotate --cleanup=verbatim --message "" --edit "$version"
    fi

# Update snapshots for all examples or specific examples
example-snapshot-update *EXAMPLES:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "{{EXAMPLES}}" ]; then
        # Update all examples
        uv run pytest tests/examples/test_example_snapshots.py --snapshot-update -v
    else
        # Update specific examples by using -k pattern matching
        for example in {{EXAMPLES}}; do
            uv run pytest tests/examples/test_example_snapshots.py -k "${example}" --snapshot-update -v
        done
    fi

# Run example snapshot tests
test-example-snapshots:
    uv run pytest tests/examples/test_example_snapshots.py -v
