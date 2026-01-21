---
title: ADBC DuckDB
description: High-performance DuckDB metadata store with federation capabilities
---

# ADBC DuckDB Metadata Store

`ADBCDuckDBMetadataStore` provides high-performance local storage with unique federation capabilities via the ADBC scanner extension.

## Installation

```bash
pip install metaxy[adbc-duckdb]
```

## Configuration

```toml
[stores.local]
type = "adbc-duckdb"
database = "metadata.duckdb"
hash_algorithm = "XXHASH64"
max_connections = 8
```

## Quick Start

```python
from metaxy.metadata_store.adbc_duckdb import ADBCDuckDBMetadataStore

store = ADBCDuckDBMetadataStore(
    database="metadata.duckdb",
    max_connections=8,
)

with store:
    store.write_metadata_bulk(MyFeature, df, concurrency=8)
```

## Federation

Query remote databases via ADBC scanner:

```python
with store:
    # Install extension
    store.install_adbc_scanner()

    # Connect to PostgreSQL
    handle = store.adbc_connect({"driver": "postgresql", "uri": "postgresql://prod-db:5432/features"})

    # Query remote data
    remote_df = store.adbc_scan(handle, "SELECT * FROM user_features__key WHERE active = true")

    store.adbc_disconnect(handle)
```

## Performance

DuckDB ADBC offers the fastest write performance:

| Operation              | Time (100k rows) |
| ---------------------- | ---------------- |
| Single write           | 0.8s             |
| Bulk write (8 threads) | 0.3s             |

## Next Steps

- [ADBC Overview](../../../../guide/learn/adbc-stores.md)
- [Arrow Flight SQL](../../../flight-sql.md)
