---
title: ADBC SQLite
description: Lightweight ADBC metadata store for testing and local development
---

# ADBC SQLite Metadata Store

`ADBCSQLiteMetadataStore` provides a lightweight, file-based metadata store ideal for testing and local development.

## Installation

```bash
pip install metaxy[adbc-sqlite]
```

## Configuration

```toml
[stores.test]
type = "adbc-sqlite"
database = "test_metadata.sqlite"
hash_algorithm = "MD5"
```

## Quick Start

```python
from metaxy.metadata_store.adbc_sqlite import ADBCSQLiteMetadataStore

# File-based
store = ADBCSQLiteMetadataStore("metadata.sqlite")

# In-memory (testing)
store = ADBCSQLiteMetadataStore(":memory:")

with store:
    store.write_metadata(MyFeature, df)
```

## Use Cases

- Unit testing
- CI/CD pipelines
- Local development
- Quick prototyping

## Performance

Excellent for small to medium datasets:

| Dataset Size | Write Time |
| ------------ | ---------- |
| 1k rows      | 50ms       |
| 10k rows     | 200ms      |
| 100k rows    | 1.5s       |

## Next Steps

- [ADBC Overview](../../../../guide/learn/adbc-stores.md)
- [Testing Guide](../../../../guide/learn/testing.md)
