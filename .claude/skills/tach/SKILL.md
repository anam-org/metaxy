---
name: tach
description: This skill should be used when the user asks to "add a tach module", "configure tach layers", "define module boundaries", "set up interfaces", "run tach check", "check module boundaries", "tach sync", "tach show", "deprecate a dependency", "tach-ignore", "unchecked modules", "tach test", "skip tests with tach", "configure tach.toml", "source roots", "forbid circular dependencies", "enforce module boundaries", "set up architectural layers", or "tach init".
---

# Tach

Tach enforces module boundaries in Python codebases. It verifies that imports between modules respect declared dependencies and public interfaces, with no runtime impact.

For full documentation: https://docs.gauge.sh/

## How This Project Uses Tach

The project defines three architectural layers in `tach.toml`: `cli` > `ext` > `core`. Higher layers may import from lower layers without declaring dependencies. Same-layer imports require explicit `depends_on` declarations. Unlayered facade and primitive modules (e.g., `metaxy`, `metaxy._decorators`) sit outside the layer hierarchy.

Run `uv run tach check` to verify boundaries. The pre-commit hook runs this automatically on `src/` changes.

Run `uv run pytest --tach` to skip tests unaffected by changes (uses tach's module dependency graph for impact analysis).

## Core Commands

| Command | Purpose |
|---------|---------|
| `tach init` | Guided setup: walks through `tach mod`, `tach sync`, `tach show` |
| `tach mod` | Interactive terminal UI to mark module boundaries |
| `tach sync` | Sync `tach.toml` with actual imports (`--add` to only add) |
| `tach check` | Report boundary/interface violations (`--exact` for unused deps) |
| `tach check-external` | Validate 3rd-party imports match `pyproject.toml` |
| `tach show` | Visualize dependency graph (`--web`, `--mermaid`, `-o`) |
| `tach map` | JSON dependency map between files (`--closure` for transitives) |
| `tach report` | Dependencies/usages of a module (`--dependencies`, `--usages`) |
| `tach test` | Run only tests impacted by changes |
| `tach install` | Install as pre-commit hook |

Full command reference: https://docs.gauge.sh/usage/commands/

## Key Concepts

### Modules

A module is a Python package or file with dependencies configured in `tach.toml`. Identified by import path from the nearest source root (e.g., `metaxy.config` for `src/metaxy/config/`).

```toml
[[modules]]
path = "metaxy.config"
layer = "core"
depends_on = ["metaxy._decorators", "metaxy.models"]
```

Special attributes:
- `utility: true` — accessible to all modules without declaring dependency
- `unchecked: true` — no dependency restrictions (for incremental adoption)
- `visibility: []` — isolate module from external imports
- `cannot_depend_on` — forbidden dependencies (takes precedence over `depends_on`)

### Layers

Ordered architectural tiers. Higher layers may freely import from lower layers; lower layers may never import from higher layers. Same-layer imports require explicit `depends_on`.

```toml
layers = ["cli", "ext", "core"]
```

Set `layers_explicit_depends_on = true` to require all cross-layer dependencies be declared explicitly. Mark a layer as closed with `{ name = "commands", closed = true }` to force higher layers through the intermediary.

Full layers documentation: https://docs.gauge.sh/usage/layers/

### Interfaces

Define public APIs to prevent deep coupling. Only imports matching `expose` patterns are allowed.

```toml
[[interfaces]]
expose = ["get_data"]
from = ["core"]
```

Interfaces support `visibility` to restrict consumers and `exclusive: true` to override other interfaces.

Full interfaces documentation: https://docs.gauge.sh/usage/interfaces/

### Deprecation

Mark dependencies as deprecated to surface usage without failing checks:

```toml
depends_on = [{ path = "core", deprecated = true }]
```

### tach-ignore

Suppress specific violations with inline comments:

```python
# tach-ignore
from core.main import private_function

from core.api import priv, pub  # tach-ignore priv
```

Add reasons: `# tach-ignore(reason here) member_name`

## Common Tasks

### Add a New Module

1. Add the module definition to `tach.toml` with `path`, `layer`, and `depends_on`
2. Run `tach sync --add` to discover any additional dependencies
3. Run `tach check` to verify

### Move a Module Between Layers

1. Update the `layer` field in `tach.toml`
2. Adjust `depends_on` based on new layer relationships
3. Run `tach check` to verify no violations

### Debug Boundary Violations

Error format: `file.py[L8]: Cannot import 'foo.bar'. Module 'baz' cannot depend on 'foo'.`

Options:
1. Add the dependency to `depends_on` in `tach.toml`
2. Mark dependency as `deprecated` to track without blocking
3. Use `# tach-ignore` for exceptions
4. Restructure imports to go through public interfaces

## Configuration Reference

Full configuration documentation: https://docs.gauge.sh/usage/configuration/
