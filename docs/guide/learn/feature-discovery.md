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

## External Features

Use [`sync_external_features`][metaxy.sync_external_features] to load feature definitions from a metadata store without requiring the original Python classes to be importable. This enables depending on features from external projects or historical snapshots where the source code is not available at runtime.

```python
import metaxy as mx

# Sync external features from the metadata store
mx.sync_external_features(store)
```

!!! note "Pydantic Schema Limitation"

    Features loaded from the metadata store have their JSON schema preserved from when they were originally saved. However, the Pydantic model class is not available. Operations that require the actual Python class, such as model instantiation or validation, will not work for these features.

Metaxy has a few safe guards in order to combat incorrect versioning information on external feature definitions. By default, Metaxy emits warnings when an external feature appears to have a different version (or field versions) than the actual feature definition loaded from the other project. These warnings can be turned into errors by:

- passing `on_conflict="raise"` to [`sync_external_features`][metaxy.sync_external_features]
- passing `--locked` to Metaxy CLI commands
- setting `locked` to `True` in the global Metaxy configuration. This can be done either in the config file or via the `METAXY_LOCKED` environment variable.

!!! tip

    We recommend setting `METAXY_LOCKED=1` in production

!!! info

    [`MetadataStore.resolve_update] always calls [`sync_external_features`][metaxy.sync_external_features] internally.
