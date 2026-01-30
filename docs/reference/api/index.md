---
title: "API Reference"
description: "Python API reference for Metaxy."
---

# API Reference

## `metaxy`

The top-level `metaxy` module provides the main public API for Metaxy. It is typically referenced as `mx`:

```python
import metaxy as mx
```

## Initialization

::: metaxy.init_metaxy

::: metaxy.sync_external_features

## Metadata Stores

Metaxy abstracts interactions with metadata through the [MetadaStore][metaxy.metadata_store.base.MetadataStore] interface.

## Dependency Specification

Metaxy has a declarative [feature specification system](./definitions/index.md) that allows users to express dependencies on versioned fields of other upstream features.
