# Recompute

> [!WARNING]
> This examples is a WIP

This example demonstrates how Metaxy automatically detects when upstream features change and recomputes downstream dependencies.

## How It Works

When a feature's `code_version` changes, Metaxy:

1. Detects the change in the feature definition
2. Identifies all downstream features that depend on it
3. Automatically recomputes those features with the new upstream data

## Plan

::: metaxy-example scenarios
example: recompute
:::

## Feature Definitions

### Initial Code

The parent feature starts with `code_version="1"`:

::: metaxy-example file
example: recompute
path: src/example_recompute/features.py
:::

### The Change

Let's change the `code_version`:

::: metaxy-example patch
example: recompute
path: patches/01_update_parent_algorithm.patch
:::

### Updated Code

::: metaxy-example file
example: recompute
path: src/example_recompute/features.py
patches: ["patches/01_update_parent_algorithm.patch"]
:::

## Key Takeaway

Metaxy ensures all features remain consistent with their dependencies. When `ParentFeature.code_version` changes from `"1"` to `"2"`, `ChildFeature` automatically recomputes—no manual tracking required.
