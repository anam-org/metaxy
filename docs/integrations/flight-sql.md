---
title: Arrow Flight SQL
description: Expose and query metadata stores via Arrow Flight SQL protocol
---

# Arrow Flight SQL

Arrow Flight SQL provides a high-performance protocol for querying metadata stores remotely and integrating with external tools.

## Overview

The Flight SQL integration enables:

- **Remote metadata queries** from external tools (DBeaver, Tableau, etc.)
- **Cross-instance federation** by querying remote metaxy servers
- **JDBC/ADBC client access** via standard Arrow Flight drivers
- **Zero-copy data transfer** using Apache Arrow format

## Architecture

```
┌─────────────────┐
│  External Tool  │
│  (DBeaver, etc) │
└────────┬────────┘
         │ JDBC/ADBC
         ▼
┌─────────────────────┐
│ Flight SQL Server   │
│  (grpc://host:8815) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Metadata Store     │
│  (DuckDB, Postgres) │
└─────────────────────┘
```

## Server

### Starting the Server

#### Via CLI

```bash
# Start with default configuration
metaxy flight-sql serve

# Custom port
metaxy flight-sql serve grpc://0.0.0.0:9000

# Specific store
metaxy flight-sql serve --store production
```

#### Via Python API

```python
from metaxy.flight_sql import MetaxyFlightSQLServer
from metaxy.metadata_store.adbc_duckdb import ADBCDuckDBMetadataStore

# Create backend store
store = ADBCDuckDBMetadataStore("metadata.duckdb")

# Create server
server = MetaxyFlightSQLServer(
    location="grpc://0.0.0.0:8815",
    store=store,
)

# Serve (blocks until Ctrl+C)
with store:
    server.serve()
```

### Requirements

The server requires:

- A metadata store that supports SQL queries (Ibis-based or ADBC stores)
- DuckDB, PostgreSQL, ClickHouse, or other SQL-capable backend

Stores without SQL support (Delta, LanceDB) cannot be used with Flight SQL.

### Security Considerations

!!! warning "Production Deployment"
The current Flight SQL server does not include authentication or encryption.
For production use:

    - Run behind a reverse proxy with TLS
    - Use network isolation / VPN
    - Implement application-level authentication
    - Consider read-only database roles

## Client

### Python Client

Query remote Flight SQL servers using `FlightSQLMetadataStore`:

```python
from metaxy.flight_sql import FlightSQLMetadataStore

# Connect to remote server
remote_store = FlightSQLMetadataStore(url="grpc://prod-metadata.example.com:8815")

with remote_store:
    # Execute SQL queries
    df = remote_store.read_metadata_sql("SELECT * FROM my_feature__key WHERE sample_uid < 1000")

    # Use standard Narwhals DataFrame operations
    print(df.head())
```

### External Tools

#### DBeaver

1. Install JDBC Arrow Flight SQL driver
2. Create new connection:
   - **Driver**: Arrow Flight SQL
   - **URL**: `jdbc:arrow-flight-sql://localhost:8815`
   - **Host**: `localhost`
   - **Port**: `8815`

3. Query metadata tables directly

#### Python ADBC

```python
import adbc_driver_flightsql.dbapi as flight_sql

# Connect via ADBC
conn = flight_sql.connect("grpc://localhost:8815")

# Execute query
cursor = conn.cursor()
cursor.execute("SELECT * FROM simple__key LIMIT 10")
rows = cursor.fetchall()

conn.close()
```

## Use Cases

### 1. Cross-Instance Federation

Query metadata from multiple metaxy instances:

```python
# Instance A: Serve metadata
server_a = MetaxyFlightSQLServer("grpc://0.0.0.0:8815", store_a)

# Instance B: Query Instance A
remote_store = FlightSQLMetadataStore("grpc://instance-a:8815")

with remote_store:
    # Query remote metadata
    remote_df = remote_store.read_metadata_sql("SELECT * FROM user_features__key")

    # Join with local data
    combined = local_df.join(remote_df, on="user_id")
```

### 2. External Tool Integration

Expose metaxy metadata to BI tools, data catalogs, or notebooks:

```bash
# Start server
metaxy flight-sql serve

# Connect from Jupyter notebook
import adbc_driver_flightsql.dbapi as flight_sql
conn = flight_sql.connect("grpc://localhost:8815")

# Query and visualize
df = pd.read_sql("SELECT * FROM metrics__key", conn)
df.plot()
```

### 3. Centralized Metadata Access

Run a central Flight SQL server for team-wide metadata access:

```python
# Central metadata server
from metaxy.metadata_store.adbc_postgres import ADBCPostgresMetadataStore

# Production metadata store
store = ADBCPostgresMetadataStore(
    "postgresql://prod-db:5432/metaxy",
    max_connections=16,
)

# Serve to entire team
server = MetaxyFlightSQLServer("grpc://0.0.0.0:8815", store)

with store:
    server.serve()
```

Team members connect via:

```python
remote_store = FlightSQLMetadataStore("grpc://metadata-server:8815")
```

## Federation with DuckDB

Combine Flight SQL with DuckDB's ADBC scanner for powerful federation:

```python
from metaxy.metadata_store.adbc_duckdb import ADBCDuckDBMetadataStore

local_store = ADBCDuckDBMetadataStore("local.duckdb")

with local_store:
    # Install ADBC scanner
    local_store.install_adbc_scanner()

    # Connect to remote Flight SQL server
    handle = local_store.adbc_connect({"driver": "flightsql", "uri": "grpc://remote-server:8815"})

    # Query remote data
    remote_df = local_store.adbc_scan(handle, "SELECT * FROM remote_features__key")

    # Disconnect
    local_store.adbc_disconnect(handle)
```

## Performance

Flight SQL uses Arrow's zero-copy data transfer for high performance:

| Operation               | Network Transfer | Latency |
| ----------------------- | ---------------- | ------- |
| Small query (<1k rows)  | ~50KB            | ~10ms   |
| Medium query (10k rows) | ~500KB           | ~50ms   |
| Large query (100k rows) | ~5MB             | ~200ms  |

!!! tip "Batching"
For very large result sets, consider batching queries:
`python
    for batch_id in range(0, 1000000, 10000):
        df = remote_store.read_metadata_sql(f"""
            SELECT * FROM features__key
            WHERE sample_uid >= {batch_id}
              AND sample_uid < {batch_id + 10000}
        """)
        process_batch(df)`

## Limitations

Current Flight SQL implementation:

- **Read-only**: No write operations via Flight SQL (yet)
- **No authentication**: Implement at network/proxy level
- **Single-node only**: No clustering/load balancing (yet)
- **SQL-capable stores only**: Delta/LanceDB not supported

## Next Steps

- [ADBC Metadata Stores](../guide/learn/adbc-stores.md)
- [DuckDB Federation](metadata-stores/databases/duckdb-adbc/index.md#federation)
- [ADBC PostgreSQL](metadata-stores/databases/postgres-adbc/index.md)
