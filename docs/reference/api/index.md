---
title: "API Reference"
description: "Python API reference for Metaxy."
---

# API Reference

## `metaxy`

The top-level `metaxy` module provides the main public API for Metaxy.

## Initialization

::: metaxy.init_metaxy

## Metadata Stores

Metaxy abstracts interactions with metadata through the [MetadaStore][metaxy.metadata_store.base.MetadataStore] interface.

## Dependency Specification

Metaxy has a declarative [feature specification system](./definitions/index.md) that allows users to express dependencies between their features and their versioned fields.
