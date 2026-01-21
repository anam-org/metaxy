---
title: ADBC PostgreSQL
description: High-performance PostgreSQL metadata store using Arrow Database Connectivity
---

# ADBC PostgreSQL Metadata Store

`ADBCPostgresMetadataStore` provides high-performance PostgreSQL storage using ADBC for 2-6x faster writes compared to the Ibis-based store.

## When to Use

Choose ADBC PostgreSQL when:

- Write performance is critical (bulk ingestion, high-frequency updates)
- Running concurrent write operations
- Production deployments requiring maximum throughput
- You need PostgreSQL's durability and ACID guarantees

Use the [Ibis PostgreSQL store](../postgres/index.md) when simplicity is preferred over performance.

## Installation

```bash
pip install metaxy[adbc-postgres]
```

## Configuration

### TOML Configuration

```toml
[stores.production]
type = "adbc-postgres"
connection_string = "postgresql://user:pass@prod-db:5432/metaxy"
hash_algorithm = "SHA256"
max_connections = 8 # Connection pool size for concurrent writes
```

### Python API

```python
from metaxy.metadata_store.adbc_postgres import ADBCPostgresMetadataStore

store = ADBCPostgresMetadataStore(
    connection_string="postgresql://localhost:5432/metaxy",
    hash_algorithm="SHA256",
    max_connections=8,
)
```

## Features

### Flattened Column Storage

ADBC stores use flattened columns for struct fields, preserving native types:

```sql
-- Traditional approach (Ibis stores)
CREATE TABLE my_feature (
    sample_id INTEGER,
    metaxy_provenance_by_field JSONB  -- All fields in JSON
);

-- ADBC approach (better performance)
CREATE TABLE my_feature (
    sample_id INTEGER,
    value TEXT,
    metaxy_provenance_by_field__value TEXT,  -- One column per field
    metaxy_provenance TEXT
);
```

Benefits:

- Native type indexing and filtering
- Better query performance
- Automatic schema evolution

### Bulk Ingestion

High-performance concurrent writes:

```python
# Prepare large dataset
metadata = store.resolve_update(MyFeature, large_df)

with store:
    # Write with 8 concurrent connections
    store.write_metadata_bulk(MyFeature, metadata, concurrency=8)
```

Performance comparison for 100k row insert:

| Method                      | Time  | Throughput   |
| --------------------------- | ----- | ------------ |
| Ibis write_metadata()       | 12.5s | 8k rows/sec  |
| ADBC write_metadata()       | 5.2s  | 19k rows/sec |
| ADBC write_metadata_bulk(8) | 2.1s  | 48k rows/sec |

### Schema Evolution

Add new fields without manual migrations:

```python
# Original feature
class UserFeature(BaseFeature, spec=FeatureSpec(key="user", fields=["email"])):
    email: str


# Add new field
class UserFeature(
    BaseFeature,
    spec=FeatureSpec(
        key="user",
        fields=["email", "phone"],  # New field
    ),
):
    email: str
    phone: str  # Automatically adds column on first write


# ADBC store executes: ALTER TABLE user ADD COLUMN metaxy_provenance_by_field__phone TEXT
```

## Connection Management

### Connection Pooling

Configure pool size based on concurrency needs:

```python
store = ADBCPostgresMetadataStore(
    connection_string="postgresql://localhost:5432/metaxy",
    max_connections=16,  # Supports up to 16 concurrent writes
)
```

### Connection String Formats

```python
# Standard format
"postgresql://user:password@host:port/database"

# With SSL
"postgresql://user:password@host:port/database?sslmode=require"

# With connection parameters
"postgresql://user:password@host:port/database?connect_timeout=10&application_name=metaxy"
```

## Performance Tuning

### PostgreSQL Server Configuration

For optimal ADBC performance:

```sql
-- postgresql.conf
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
max_connections = 100
```

### Batch Size Tuning

Larger batches reduce overhead but increase memory:

```python
# Default: writes entire DataFrame
store.write_metadata(feature, large_df)

# For very large datasets, partition manually
chunk_size = 50000
for i in range(0, len(large_df), chunk_size):
    chunk = large_df[i : i + chunk_size]
    metadata = store.resolve_update(feature, chunk)
    store.write_metadata(feature, metadata)
```

## Production Considerations

### High Availability

Use PostgreSQL replication for HA:

```toml
# Write to primary
[stores.primary]
type = "adbc-postgres"
connection_string = "postgresql://user:pass@primary:5432/metaxy"

# Read from replica
[stores.replica]
type = "adbc-postgres"
connection_string = "postgresql://user:pass@replica:5432/metaxy"
```

### Monitoring

Monitor ADBC connection pool usage:

```python
# Connection metrics available in PostgreSQL
SELECT * FROM pg_stat_activity WHERE application_name = 'metaxy';
```

### Backup & Recovery

ADBC stores use standard PostgreSQL tables:

```bash
# Backup
pg_dump metaxy > metaxy_backup.sql

# Restore
psql metaxy < metaxy_backup.sql
```

## Migration from Ibis

ADBC and Ibis PostgreSQL stores are compatible:

```python
# Write with Ibis
from metaxy.metadata_store.postgres import PostgresMetadataStore

ibis_store = PostgresMetadataStore("postgresql://localhost/metaxy")

with ibis_store:
    ibis_store.write_metadata(MyFeature, df)

# Read with ADBC (same tables)
from metaxy.metadata_store.adbc_postgres import ADBCPostgresMetadataStore

adbc_store = ADBCPostgresMetadataStore("postgresql://localhost/metaxy")

with adbc_store:
    metadata = adbc_store.read_metadata(MyFeature)
```

Update configuration to switch:

```diff
 [stores.production]
-type = "postgres"
+type = "adbc-postgres"
 connection_string = "postgresql://..."
+max_connections = 8
```

## Limitations

- PostgreSQL 12+ required
- No MotherDuck/cloud-specific features
- Requires ADBC driver installation

## Next Steps

- [ADBC Overview](../../../../guide/learn/adbc-stores.md)
- [Bulk Ingestion Guide](../../../../guide/learn/adbc-stores.md#bulk-ingestion-api)
- [Arrow Flight SQL](../../../flight-sql.md)
