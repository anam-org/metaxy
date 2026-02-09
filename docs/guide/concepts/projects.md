---
title: "Metaxy Projects"
description: "Learn about Projects in Metaxy and setup multi-project environments"
---

# Metaxy Projects

As the data processing pipeline grows, it often becomes necessary to split it into multiple Python projects.

Metaxy has a Project concept that allows users to do exactly that.

## Metaxy Project

A Metaxy project is a collection of features defined in the same location: typically a Python package. By default, each Metaxy feature definition is assigned a project name based on the top-level Python module of the feature definition.

??? example "Project Name Inference"

    <!-- skip next -->
    ```py title="my_package.my_feature.py"
    import metaxy as mx

    class NewFeature(mx.BaseFeature, spec=mx.FeatureSpec(key="new/feature", id_columns=["id"])):
        id: str

    assert mx.get_feature_by_key("new/feature").project == "my_package"
    ```

As long as all feature definitions are located in the same project, users won't need to interact with the project concept when using Metaxy.

Once the codebase is split into multiple projects, certain Metaxy operations start to operate at the scope of a specific project. Users will need to configure the current Metaxy project using the `MetaxyConfig.project` setting when using the Metaxy CLI or setting up feature dependencies across separate Python environments, typically via the config file:

=== "metaxy.toml"
    ```toml
    project = "my_package"
    ```
=== "pyproject.toml"
    ```toml
    [tool.metaxy]
    project = "my_package"
    ```
=== "Environment Variable"
    ```bash
    METAXY_PROJECT=my_package
    ```

Metaxy has a feature discovery mechanism that automatically loads feature definitions from other Metaxy projects.

Certain Metaxy CLI actions can only be performed on a specific project: for example, `metaxy push` must specify (1) the project name used to select which feature definitions to serialize into the metadata store.
{ .annotate }

1. or be able to infer

## Multiple Python Environments

!!! info "Advanced Concept"

    Multi-environment setups are an advanced way of organizing Metaxy feature definitions and are not needed in most scenarios.

Metaxy supports splitting feature definitions across distinct Python environments. This is useful when different projects require very different - often incompatible - Python dependencies, and features cannot be imported at runtime. This is achieved by adding [external features](./definitions/features.md#external-features) to the Metaxy feature graph.

To avoid having to maintain these external feature definitions in sync with the actual feature definitions from the external project manually, Metaxy provides a CLI command to pull these feature definitions from the metadata store (where they should be pushed in advance):

```shell
metaxy lock
```

This command will analyze the current Metaxy feature graph and attempt to load feature definitions for unresolved dependencies from the metadata store. These feature definitions will be serialized to a `metaxy.lock` file. [`metaxy.init`][metaxy.init] will then automatically add these feature definitions to the feature graph going forward.

Users are expected to update the lock file manually when needed. By default, Metaxy will detect outdated external features in the lock file and emit warnings (or errors if configured otherwise) at runtime. Learn more about staleness detection [here](./definitions/features.md#outdated-external-features).

Here is an example of how a multi-environment setup may work with Metaxy:

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

Features from both projects can freely depend on each other. For example:

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

Then, the following sequence:

```shell
cd project_a && metaxy push
cd project_b && metaxy lock
```

will populate `project_b/metaxy.lock` with feature definition from `project_a`.
