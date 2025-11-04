# Feature Discovery

Metaxy provides automatic feature discovery through Python's entrypoint system. This enables modular architecture patterns essential for scaling Metaxy projects.

## Why Feature Discovery?

Manual feature registration doesn't scale. As your system grows, you need:

- **Plugin architectures** - Third-party teams contribute features without modifying core code
- **Feature collections** - Package and distribute related features as installable units
- **Monorepo support** - Discover features across multiple packages in a monorepo
- **Internal packages** - Share features between projects via private package registries

Feature discovery solves these problems through automatic registration at import time.

## Package Entry Points

The most powerful discovery mechanism uses Python's standard entry point system via a well-known `"metaxy.project"` entrypoint group in the package metadata.

### Creating a Feature Plugin

Structure your feature package:

```
my-video-features/
├── pyproject.toml
└── src/
    └── my_video_features/
        ├── __init__.py
        ├── detection.py
        └── transcription.py
```

Declare entry points in `pyproject.toml`:

```toml
[project]
name = "my-video-features"
version = "1.0.0"
dependencies = ["metaxy"]

[project.entry-points."metaxy.project"]
my-video-features = "my_video_features"
```

The entry point name is your project name. The value can be either:

- **Function syntax** (`module:function`) - Points to a callable function that will be invoked to load features. Useful when you need conditional loading or setup logic.
- **Module syntax** (`module`) - Points directly to a module containing Feature definitions. Simply importing the module registers the features.

!!! warning "One Entry Point Per Package"

    Each package can only declare **one** entry point in the `metaxy.project` group, since `metaxy.toml` only supports a single `project` field.

    To organize features into logical groups within a package, use submodules and import them from your entry point function.

### Installing and Using Feature Plugins

Install the package:

```bash
pip install my-video-features
# Or in a monorepo:
pip install -e ./packages/my-video-features
```

!!! warning "UV Package Manager: Entry Point Changes"

    If you're using `uv` and modify entry points in `pyproject.toml`, `uv sync` will **not** recreate the editable package metadata. You must explicitly reinstall:

    ```bash
    uv sync --reinstall-package my-video-features my-video-features
    ```

## Monorepo Patterns

In monorepos, use entry points to manage feature collections across teams:

### Team-Owned Feature Packages

```
monorepo/
├── packages/
│   ├── core-features/
│   │   └── pyproject.toml  # [project.entry-points."metaxy.features"]
│   ├── ml-features/
│   │   └── pyproject.toml  # [project.entry-points."metaxy.features"]
│   └── experimental-features/
│       └── pyproject.toml  # [project.entry-points."metaxy.features"]
└── apps/
    └── main-pipeline/
        └── pyproject.toml  # depends on feature packages
```

Each team maintains their features independently:

```toml
# packages/ml-features/pyproject.toml
[project.entry-points."metaxy.project"]
ml-features = "ml_features.load"
```

```toml
# packages/core-features/pyproject.toml
[project.entry-points."metaxy.project"]
core-features = "core_features.load"
```

The main application imports features from all installed packages, and each feature automatically knows its project based on the entry point.

## Config-Based Discovery

For simpler use cases that don't require distribution, you can specify module paths directly in configuration:

=== "metaxy.toml"

    ```toml
    project = "my-project"
    entrypoints = [
        "myapp.features.video",
        "myapp.features.audio",
    ]
    ```

=== "pyproject.toml"

    ```toml
    [tool.metaxy]
    project = "my-project"
    entrypoints = [
        "myapp.features.video",
        "myapp.features.audio",
    ]
    ```

## Best Practices

1. **Use entry points for distribution** - Any features intended for reuse should use entry points
2. **Version your feature packages** - Use semantic versioning for feature collections
3. **Test in isolation** - Load feature packages into test graphs to verify behavior

The entry point system transforms feature management from a manual process to an automatic, scalable system that grows with your organization.
