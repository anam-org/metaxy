---
title: "Feature Discovery"
description: "Automatic feature discovery from Python modules."
---

# Feature Discovery

!!! warning

    This page is WIP

## Config-Based Discovery

Specify paths for modules containing Metaxy features in Metaxy configuration:

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

## Loading from Metadata Store

Use [`load_feature_definitions`][metaxy.load_feature_definitions] to load feature definitions from a metadata store without requiring the original Python classes to be importable. This enables depending on features from external projects or historical snapshots where the source code is not available at runtime.

```python
import metaxy as mx

# Load all features from latest snapshots into active graph
definitions = mx.load_feature_definitions(store)

# Load from a specific project into a custom graph
with mx.FeatureGraph().use():
    definitions = mx.load_feature_definitions(store, projects="external-project")
```

!!! note "Pydantic Schema Limitation"

    Features loaded from the metadata store have their JSON schema preserved from when they were originally saved. However, the Pydantic model class is not available. Operations that require the actual Python class, such as model instantiation or validation, will not work for these features.
