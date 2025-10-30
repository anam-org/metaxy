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

The most powerful discovery mechanism uses Python's standard entry point system via `project.entry-points."metaxy.features"` in `pyproject.toml`.

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

[project.entry-points."metaxy.features"]
detection = "my_video_features.detection"
transcription = "my_video_features.transcription"
```

Define features in the modules:

```python
# my_video_features/detection.py
from metaxy import Feature, FeatureSpec, FeatureKey


class FaceDetection(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["video", "face_detection"]),
        # ... spec details
    ),
):
    pass


class ObjectDetection(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["video", "object_detection"]),
        # ... spec details
    ),
):
    pass
```

### Installing and Using Plugins

Install the package:

```bash
pip install my-video-features
# Or in a monorepo:
pip install -e ./packages/my-video-features
```

!!! warning "UV Package Manager: Entry Point Changes"

    If you're using `uv` and modify entry points in `pyproject.toml`, `uv sync` will **not** recreate the editable package metadata. You must explicitly reinstall:

    ```bash
    uv sync --reinstall-package my-video-features
    ```

Installed features will be automatically discovered and loaded:

```python
from metaxy import init_metaxy

# Automatically discovers and loads all features
# from installed packages with metaxy.features entry points
init_metaxy()

# Features are now available in the global graph
from metaxy import FeatureGraph

graph = FeatureGraph.get_active()
print(f"Loaded {len(graph.features_by_key)} features")
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
[project.entry-points."metaxy.features"]
embeddings = "ml_features.embeddings"
classifiers = "ml_features.classifiers"
```

The main application discovers all features.

## Config-Based Discovery

For simpler use cases, load features directly from module paths specified in Metaxy configuration:

=== "metaxy.toml"

    ```toml
    entrypoints = [
        "myapp.features.video",
    ]
    ```

=== "pyproject.toml"

    ```toml
    [tool.metaxy]
    entrypoints = [
        "myapp.features.video",
    ]
    ```

```python
from metaxy import init_metaxy

# Discovers features from configured entrypoints
init_metaxy()
```

## Best Practices

1. **Use entry points for distribution** - Any features intended for reuse should use entry points
2. **Version your feature packages** - Use semantic versioning for feature collections
3. **Test in isolation** - Load feature packages into test graphs to verify behavior

The entry point system transforms feature management from a manual process to an automatic, scalable system that grows with your organization.
