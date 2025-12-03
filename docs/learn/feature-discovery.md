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
