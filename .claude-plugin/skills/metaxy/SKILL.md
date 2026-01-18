---
name: metaxy
description: Use when working with Metaxy, a feature metadata management system for multi-modal data and ML pipelines. Helps with feature definitions, versioning, metadata stores, and testing.
---

# Metaxy

Metaxy is a metadata layer for multi-modal Data and ML pipelines that manages and tracks feature versions, dependencies, and data lineage across complex computational graphs.

## Core Concepts

- **Feature Definitions**: Declarative Python classes that define metadata schemas with field-level dependencies
- **Data Versioning**: Automatic tracking of sample versions with change propagation
- **Metadata Stores**: Pluggable backends (DuckDB, ClickHouse, BigQuery, LanceDB, Delta Lake) for storing feature metadata
- **Feature Graph**: Dependency graph of features with partial data dependency tracking

## Key Capabilities

1. **Incremental Processing**: Skip downstream updates when only unrelated fields change
2. **Field-Level Dependencies**: Express partial data dependencies to avoid unnecessary recomputations
3. **Backend Agnostic**: Works with any tabular compute engine via Narwhals (Polars, Pandas, Spark)

## Documentation

For comprehensive documentation, visit: https://anam-org.github.io/metaxy/

Key documentation pages:

- **Quickstart**: https://anam-org.github.io/metaxy/guide/overview/quickstart/
- **Feature Definitions**: https://anam-org.github.io/metaxy/guide/learn/feature-definitions/
- **Data Versioning**: https://anam-org.github.io/metaxy/guide/learn/data-versioning/
- **Metadata Stores**: https://anam-org.github.io/metaxy/guide/learn/metadata-stores/
- **Integrations**: https://anam-org.github.io/metaxy/integrations/
- **API Reference**: https://anam-org.github.io/metaxy/reference/api/
