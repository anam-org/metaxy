---
title: "Metaxy Projects"
description: "Learn about Metaxy Projects and how to organize Metaxy features into separate Python packages."
---

# Metaxy Projects

As the data processing pipeline grows, it often becomes necessary to split it into multiple Python projects.

Metaxy has a Project system which helps with organizing Metaxy features into separate Python packages.

## Metaxy Project

A Metaxy project is a collection of features defined in the same location: typically a Python package. By default, each Metaxy feature definition is assigned a project name based on the top-level Python module where it's defined in.

??? example "Project Name Inference"

    <!-- skip next -->
    ```py title="my_package.my_feature.py"
    import metaxy as mx

    class NewFeature(mx.BaseFeature, spec=mx.FeatureSpec(key="new/feature", id_columns=["id"])):
        id: str

    assert mx.get_feature_by_key("new/feature").project == "my_package"
    ```

As long as all feature definitions are located in the same project, users won't need to interact with the project concept when using Metaxy, and a single global project exists implicitly.

Once the codebase is split into multiple projects, certain Metaxy operations start to operate at the scope of a specific project. Users then need to explicitly set the current Metaxy project by using the `MetaxyConfig.project`, typically via the [config file](/reference/configuration.md):

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

Some Metaxy CLI commands can only be executed within a specific project: for example, `metaxy push` must specify (1) the project to be serialized.
{ .annotate }

1. or be able to infer

## Feature Discovery

Because features can depend on features from other projects, it becomes necessary to register all the necessary feature definitions from different projects on the same global [feature graph][metaxy.FeatureGraph] at runtime.

There are two ways for Metaxy to register a feature definition:

1. From a Python class that inherits from [`BaseFeature`][metaxy.BaseFeature]. The feature definitions is registered in Metaxy as soon as the class is created.

2. From a `metaxy.lock` file. This is advanced functionality only needed when working with [external features](./definitions/external-features.md). This happens automatically when calling [`metaxy.init`][metaxy.init] and doesn't require any additional setup.

To assist with step (1) and lift the burden of manually importing all the required features, Metaxy provides two options to automate this process: [config entry points](#config-entry-points) and [distribution entry points](#distribution-entry-points).

### Config Entry Points

Module paths with Metaxy features can be specified in the Metaxy config:

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

These entrypoints only take effect in the current Metaxy project.

### Distribution Entry Points

Metaxy also supports automatically exposing feature definitions to other packages in the same Python environment.
This can be achieved by setting `"metaxy.project"` [distribution entry point](https://packaging.python.org/en/latest/specifications/entry-points/). For example:

```toml title="pyproject.toml"
[project.entry-points."metaxy.project"]
my-key = "my_package.features"
```

!!! note

    Currently the name of the key (`my-key` in the example above) is not used by Metaxy and is not important.
