# Entrypoint System

Metaxy provides a flexible entrypoint system for automatically discovering and loading Feature classes from your project and plugins.

## Overview

The entrypoint system supports two approaches:

1. **Config-based entrypoints**: Direct module imports from a list
2. **Package-based entrypoints**: Standard Python entry points via `importlib.metadata`

Both approaches automatically register Features to the active `FeatureGraph` when modules are imported.

## Config-Based Entrypoints

Simplest approach for application code and local development.

### Usage

```python
from metaxy import load_config_entrypoints

# Load features from specific modules
load_config_entrypoints([
    "myapp.features.video",
    "myapp.features.audio",
    "myapp.features.text"
])

# All features from these modules are now registered
```

### Example Feature Module

```python
# myapp/features/video.py
from metaxy import Feature, FeatureSpec, FeatureKey, FieldSpec, FieldKey

class VideoFrames(Feature, spec=FeatureSpec(
    key=FeatureKey(["video", "frames"]),
    deps=None,
    fields=[
        FieldSpec(key=FieldKey(["default"]), code_version=1)
    ]
)):
    """Video frame features - auto-registered on import."""
    pass

class VideoMetadata(Feature, spec=FeatureSpec(
    key=FeatureKey(["video", "metadata"]),
    deps=None,
    fields=[
        FieldSpec(key=FieldKey(["default"]), code_version=1)
    ]
)):
    """Video metadata features - auto-registered on import."""
    pass
```

## Package-Based Entrypoints

Standard Python packaging approach for dependencies, plugins and third-party extensions.

### Declaring Entry Points

In your package's `pyproject.toml`:

```toml
[project.entry-points."metaxy.features"]
video = "myplugin.features.video"
audio = "myplugin.features.audio"
```

### Discovering Entry Points

```python
from metaxy import load_package_entrypoints

# Discover and load all installed plugins
load_package_entrypoints()

# Or use a custom entry point group
load_package_entrypoints(group="myapp.plugins")
```

## Combined Discovery

Load from both config and packages:

```python
from metaxy import load_features

# Load from both sources
graph = load_features(
    config_entrypoints=[
        "myapp.features.core",
        "myapp.features.custom"
    ],
    load_packages=True
)

print(f"Loaded {len(graph.features_by_key)} features")
```

## Advanced Usage

### Custom Registry

Load features into a custom graph instead of the global one (for example, for testing):

```python
from metaxy import FeatureGraph, load_config_entrypoints

# Create isolated graph
test_graph = FeatureGraph()

with test_graph.use():
    load_config_entrypoints([
        "myapp.test_features"
    ])

# Features are in test_graph, not the global graph
```

### Single Module Loading

Load a single module entrypoint:

```python
from metaxy import load_module_entrypoint

load_module_entrypoint("myapp.features.video")
```
