# Metaxy

## Overview

Metaxy is a feature metadata management system that tracks feature versions, dependencies, and data lineage for multimodal pipelines. It enables:

- **Declarative Feature Definitions**: Define features with explicit dependencies and versioning
- **Automatic Change Detection**: Track when feature code changes and identify affected downstream features
- **Smart Migrations**: Reconcile metadata when code refactors don't change computation (avoiding unnecessary recomputation)
- **Dependency-Aware Updates**: Automatically recompute features when upstream dependencies actually change
- **Immutable Metadata**: Copy-on-write metadata store preserves historical versions
- **Graph Snapshots**: Record complete feature graph states in your deployment pipeline

### Key Concepts

- **Feature Version**: Deterministic hash of feature definition (dependencies, fields, code versions)
- **Data Version**: Hash of upstream data versions - automatically invalidates when dependencies change
- **Migrations**: Explicit operations to update metadata when features are refactored without changing outputs
- **Recomputation**: Automatic when actual computation logic changes (detected via code_version bumps)

## Examples

See [examples](examples/README.md).
