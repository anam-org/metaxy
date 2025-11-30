# Feature Discovery

In Metaxy, feature definitions are associated with Metaxy projects.
Feature definitions may come from two sources:

- all feature classes from the current Python project

- feature definitions previously pushed to the metadata store (the project they belong to has been recorded at push time)

!!! tip

    To push your project, use `metaxy graph push` CLI

This means that the Metaxy project name should always match the Python project name.
