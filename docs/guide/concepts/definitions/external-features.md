---
title: "External Features"
description: "Declarative feature definitions with Pydantic models."
---

## External Features

!!! info "Advanced Concept"

    External features are an advanced way of organizing Metaxy feature definitions. It's only needed when depending on features from separate Python environments.

External features are stubs pointing at features actually defined in other Metaxy projects and not available at runtime.
External feature definitions are stored in a `metaxy.lock` file and are loaded automatically when [`metaxy.init`][metaxy.init] is called.

??? info annotate "Manually Registering External Features"

      Users who need more control or cannot use the `metaxy.lock` file for some reasons (1) can create them manually:

      ```py
      import metaxy as mx

      external_feature = mx.FeatureDefinition.external(
          spec=mx.FeatureSpec(key="a/b/c", id_columns=["id"]),
          project="external-project",
      )

      mx.FeatureGraph.get().add_feature_definition(external_feature)
      ```

1. Please let us know why via a [GitHub Issue](https://github.com/anam-org/metaxy/issues/new)

### `metaxy.lock` file

Metaxy can automatically generate a `metaxy.lock` file with external feature definitions from other Metaxy projects. In order to do this:

1. Run `metaxy push` in other Metaxy projects - this will serialize feature definitions to the metadata store.

    !!! tip
        This command is expected to be executed as part of the CI/CD pipeline.

2. Run `metaxy lock` in the current Metaxy project - this pulls external feature definitions from the metadata store into a `metaxy.lock` file. Only explicit feature dependencies are pulled in (plus any [`extra_features`](#loading-extra-features) configured in `metaxy.toml`). Subsequent [`metaxy.init`][metaxy.init] calls will now automatically add these feature definitions to the feature graph.

!!! example "Multi-environment setup"

    ```
    root/
    ├── project_a/
    │   ├── .venv/
    │   ├── metaxy.toml
    │   ├── metaxy.lock
    │   └── features.py
    └── project_b/
        ├── .venv/
        ├── metaxy.toml
        ├── metaxy.lock
        └── features.py
    ```

    Features from both projects can freely depend on each other:

    <!-- skip next -->
    ```py title="project_b/features.py"
    import metaxy as mx

    class FeatureB(
        mx.BaseFeature,
        spec=mx.FeatureSpec(
            id_columns=["id"],
            deps=["feature/from/project/a"]
        )
    ):
        id: str

    ```

    Then, the following commands:

    ```shell
    cd project_a && metaxy push
    cd project_b && metaxy lock
    ```

    will populate `project_b/metaxy.lock` with feature definition from `project_a`.

### Syncing External Features

External features only exist on the feature graph until actual feature definitions are loaded from the metadata store to replace them. This can be done with [`metaxy.sync_external_features`][metaxy.sync_external_features].

```python
import metaxy as mx

# Sync external features from the metadata store
mx.sync_external_features(store)
```

!!! note "Pydantic Schema Limitation"

    Features loaded from the metadata store have their JSON schema preserved from when they were originally saved. However, the Python *class* may not be available anymore. Operations that require the actual Python class, such as model instantiation or validation, will not work for these features.

Users are expected to keep the lock file up to date, but Metaxy has a few safe guards to protect users from using stale external features. If the actual feature pulled from the metadata store has a different version than the one in the lock file, `sync_external_features` emits warnings. An exception can be raised instead by:

- passing `on_conflict="raise"` to `sync_external_features`

- passing `--locked` to Metaxy CLI commands

- setting `locked` to `True` in the global Metaxy configuration. This can be done either in the config file or via the `METAXY_LOCKED` environment variable.

!!! tip
    We strongly recommend setting `METAXY_LOCKED=1` in production

Additionally, the following actions always trigger `sync_external_features`:

- pushing feature definitions to the metadata store (e.g. `metaxy push` CLI)

- [`MetadataStore.read`][metaxy.MetadataStore.resolve_update]

- [`MetadataStore.resolve_update`][metaxy.MetadataStore.resolve_update]

!!! tip "Disabling Automatic Syncing"

    This behavior can be disabled by setting `sync=False` in the global Metaxy configuration. However, we advise to keep it enabled,
    because [`sync_external_features`][metaxy.sync_external_features] is very lightweight on the first call and a no-op on subsequent calls.
    It only does anything if the current feature graph does not contain any external features.

### Loading Extra Features

Sometimes a project needs access to feature definitions it does not directly depend on. For example, a dashboard service might display metadata for features produced by other projects without computing any features itself. Since these feature definitions are not referenced via `deps=`, they would not normally be pulled into the graph.

#### Via `sync_external_features`

The first option is to do it manually by providing a [`FeatureSelection`][metaxy.FeatureSelection] to `sync_external_features`.

#### Via Metaxy configuration

The `[[extra_features]]` section in `metaxy.toml` can be used to declare additional feature definitions to load from the metadata store. Each entry is a [`FeatureSelection`][metaxy.FeatureSelection].

```toml title="metaxy.toml"
project = "my_dashboard"

[[extra_features]]
projects = ["team_a"]

[[extra_features]]
projects = ["team_b"]
keys = ["shared/feature_x"]
```

To load every feature available in the metadata store:

```toml title="metaxy.toml"
[[extra_features]]
all = true
```

Multiple entries are combined together. The resulting selection is used automatically by [`sync_external_features`][metaxy.sync_external_features], `metaxy lock`, and any operation that triggers a sync.
