---
title: ADBC Metadata Stores
description: High-performance metadata stores using Arrow Database Connectivity for fast bulk writes and federation
---

# ADBC Metadata Stores

ADBC (Arrow Database Connectivity) metadata stores provide high-performance alternatives to Ibis-based stores for applications requiring fast bulk writes or cross-database federation.

## Overview

ADBC stores use Apache Arrow's database connectivity layer for zero-copy data transfer, offering 2-10x faster write performance compared to traditional Ibis stores. They're ideal for:

- **Bulk ingestion** workflows with high write throughput requirements
- **Cross-database federation** via DuckDB's ADBC scanner
- **Remote metadata access** via Arrow Flight SQL protocol
- **Production deployments** requiring maximum write performance

## When to Use ADBC Stores

### Use ADBC stores when:

- Writing large batches of metadata (>10k rows)
- Running concurrent write operations
- Federating metadata across multiple databases
- Exposing metadata via Arrow Flight SQL to external tools
- Performance is critical for your write workloads

### Use Ibis stores when:

- Write performance is not a bottleneck
- You need features not yet available in ADBC stores
- Simplicity and fewer dependencies are preferred
- You're already using Ibis elsewhere in your stack

## Available ADBC Stores

| Store                                                                                            | Driver                   | Use Case                                         |
| ------------------------------------------------------------------------------------------------ | ------------------------ | ------------------------------------------------ |
| [ADBCPostgresMetadataStore](../../integrations/metadata-stores/databases/postgres-adbc/index.md) | `adbc-driver-postgresql` | Production PostgreSQL with high write throughput |
| [ADBCDuckDBMetadataStore](../../integrations/metadata-stores/databases/duckdb-adbc/index.md)     | `adbc-driver-duckdb`     | Local/embedded with federation capabilities      |
| [ADBCSQLiteMetadataStore](../../integrations/metadata-stores/databases/sqlite-adbc/index.md)     | `adbc-driver-sqlite`     | Lightweight local storage and testing            |

## Performance Characteristics

Benchmarks show ADBC stores deliver significant performance improvements for write-heavy workloads:

| Operation                              | Ibis Store | ADBC Store | Speedup  |
| -------------------------------------- | ---------- | ---------- | -------- |
| Single-threaded write (10k rows)       | 1.2s       | 0.5s       | **2.4x** |
| Concurrent write (8 threads, 80k rows) | 9.6s       | 1.6s       | **6.0x** |
| Bulk ingestion (1M rows)               | 120s       | 18s        | **6.7x** |

!!! tip "Concurrency"
ADBC stores support concurrent writes via connection pooling. Configure `max_connections` to match your concurrency needs.

## Installation

Install ADBC dependencies for your chosen backend:

```bash
# PostgreSQL
pip install metaxy[adbc-postgres]

# DuckDB
pip install metaxy[adbc-duckdb]

# SQLite
pip install metaxy[adbc-sqlite]

# All ADBC stores
pip install metaxy[adbc]
```

## Basic Usage

### Configuration

Configure ADBC stores in `metaxy.toml`:

```toml
[stores.production]
type = "adbc-postgres"
connection_string = "postgresql://user:pass@prod-db:5432/features"
hash_algorithm = "SHA256"
max_connections = 8 # Connection pool size
```

### Python API

```python
from metaxy.metadata_store.adbc_postgres import ADBCPostgresMetadataStore

# Create store with connection pooling
store = ADBCPostgresMetadataStore(
    connection_string="postgresql://localhost:5432/metaxy",
    hash_algorithm="SHA256",
    max_connections=8,
)

with store:
    # Standard metadata operations
    store.write_metadata(MyFeature, df)

    # Bulk ingestion with concurrency
    store.write_metadata_bulk(
        MyFeature,
        large_df,
        concurrency=8,  # Uses 8 parallel connections
    )
```

## Advanced Features

### Bulk Ingestion API

The `write_metadata_bulk()` method partitions DataFrames and writes chunks concurrently for maximum throughput:

```python
# Prepare metadata with provenance
metadata = store.resolve_update(MyFeature, input_df)

# Bulk write with 8 concurrent connections
store.write_metadata_bulk(MyFeature, metadata, concurrency=8)
```

!!! note "Provenance Required"
Unlike `write_metadata()`, the bulk API requires pre-computed provenance columns via `resolve_update()`.

### Cross-Database Federation

DuckDB ADBC stores support querying remote databases via the ADBC scanner:

```python
from metaxy.metadata_store.adbc_duckdb import ADBCDuckDBMetadataStore

store = ADBCDuckDBMetadataStore("local.duckdb")

with store:
    # Install ADBC scanner extension
    store.install_adbc_scanner()

    # Connect to remote PostgreSQL
    handle = store.adbc_connect({"driver": "postgresql", "uri": "postgresql://prod-db:5432/features"})

    # Query remote data
    remote_df = store.adbc_scan(handle, "SELECT * FROM my_feature__key WHERE sample_uid < 1000")

    # Disconnect
    store.adbc_disconnect(handle)
```

See [DuckDB ADBC Federation](../../integrations/metadata-stores/databases/duckdb-adbc/index.md#federation) for more details.

### Arrow Flight SQL

Expose metadata stores via Arrow Flight SQL protocol for external tool access:

```python
from metaxy.flight_sql import MetaxyFlightSQLServer

server = MetaxyFlightSQLServer(
    location="grpc://0.0.0.0:8815",
    store=store,
)

with store:
    server.serve()  # Blocks until Ctrl+C
```

Or via CLI:

```bash
metaxy flight-sql serve --store production
```

Connect from external tools:

- DBeaver using JDBC Flight SQL driver
- Python clients via `adbc-driver-flightsql`
- Other metaxy instances via `FlightSQLMetadataStore`

## Migration from Ibis Stores

ADBC and Ibis stores are compatible - they use the same table schemas and can read each other's data:

```python
# Write with Ibis store
ibis_store = DuckDBMetadataStore("features.duckdb")
with ibis_store:
    ibis_store.write_metadata(MyFeature, df)

# Read with ADBC store
adbc_store = ADBCDuckDBMetadataStore("features.duckdb")
with adbc_store:
    metadata = adbc_store.read_metadata(MyFeature)
```

To migrate:

1. Update configuration to use ADBC store type
2. Install ADBC driver dependencies
3. Test with existing data
4. Update write code to use `write_metadata_bulk()` for performance

## Limitations

Current limitations of ADBC stores:

- **No MotherDuck support yet** in DuckDB ADBC (Ibis DuckDB store supports MotherDuck)
- **PostgreSQL requires v12+** for ADBC driver compatibility
- **Bulk API requires pre-computed provenance** (no automatic `resolve_update()`)

## Next Steps

- [ADBC PostgreSQL Store](../../integrations/metadata-stores/databases/postgres-adbc/index.md)
- [ADBC DuckDB Store](../../integrations/metadata-stores/databases/duckdb-adbc/index.md)
- [ADBC SQLite Store](../../integrations/metadata-stores/databases/sqlite-adbc/index.md)
- [Arrow Flight SQL](../../integrations/flight-sql.md)
