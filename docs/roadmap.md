# Metaxy Roadmap

This document outlines planned features and enhancements for Metaxy.

## MaterializationCatalog

### Motivation

Currently, checking if a feature exists requires attempting to read from storage (error-handling approach). For large-scale deployments with many features and storage backends, this can be inefficient.

A **MaterializationCatalog** provides a lightweight index for fast existence checks without touching the actual data storage.

### Design

```python
class MaterializationCatalog(ABC):
    """
    Optional catalog for tracking materialized features.
    
    This is metadata about metadata - a lightweight index for fast lookups.
    Completely optional - MetadataStore works fine without it.
    """
    
    @abstractmethod
    def register_materialization(
        self,
        feature: FeatureKey | type[Feature],
        df: pl.DataFrame,
    ) -> None:
        """
        Register that a feature has been materialized.
        
        Extracts:
        - Container keys from data_version struct
        - Data versions (unique values)
        - Sample counts
        - Timestamp
        """
        pass
    
    @abstractmethod
    def has_feature(
        self,
        feature: FeatureKey | type[Feature],
        *,
        container: ContainerKey | None = None,
        data_version: str | None = None,
    ) -> bool:
        """
        Check if data exists without reading it.
        
        Args:
            container: Check specific container (None = any container)
            data_version: Check specific version (None = any version)
        """
        pass
    
    @abstractmethod
    def get_materialized_versions(
        self,
        feature: FeatureKey | type[Feature],
        container: ContainerKey | None = None,
    ) -> set[str]:
        """Get all data versions that have been materialized."""
        pass
    
    @abstractmethod
    def list_features(self) -> list[FeatureKey]:
        """List all features in catalog."""
        pass
    
    def get_materialization_info(
        self,
        feature: FeatureKey | type[Feature],
    ) -> pl.DataFrame:
        """
        Get detailed materialization info.
        
        Returns DataFrame with columns:
        - container_key
        - data_version
        - sample_count
        - materialized_at
        """
        pass
```

### Usage

```python
# Create catalog
catalog = SQLiteCatalog(db_path="/data/catalog.db")

# Create store with catalog
store = DeltaMetadataStore(
    table_uri="s3://bucket/metadata",
    catalog=catalog  # Optional
)

# Catalog is automatically updated on writes
store.write_metadata(MyFeature, df)

# Fast existence checks (no data access)
if catalog.has_feature(MyFeature):
    df = store.read_metadata(MyFeature)

# Query what's materialized
materialized_features = catalog.list_features()
versions = catalog.get_materialized_versions(MyFeature, container="default")
```

### Implementation Priority

- **Phase 1**: `InMemoryCatalog` - Simple dict-based, single process
- **Phase 2**: `SQLiteCatalog` - Persistent, file-based, good for concurrent access
- **Phase 3**: `PostgresCatalog` - Production-grade, multi-user environments

### Benefits

- **Performance**: Fast existence checks without storage I/O
- **Observability**: Track what's materialized across environments
- **Audit trail**: History of materializations with timestamps
- **Query interface**: Find features by various criteria

### Trade-offs

- **Complexity**: Additional component to manage
- **Consistency**: Catalog can drift from actual storage
- **Optional**: Store works fine without it (graceful degradation)

## Additional Storage Backends

### Delta Lake Backend

Production-grade storage with advanced features:

```python
class DeltaMetadataStore(MetadataStore):
    """
    Delta Lake storage backend.
    
    Features:
    - ACID transactions
    - Time travel (query historical versions)
    - Schema evolution
    - Efficient upserts
    - S3/GCS/Azure support
    """
    
    def __init__(
        self,
        table_uri: str,
        *,
        partition_cols: list[str] | None = None,
        storage_options: dict | None = None,
        **kwargs
    ):
        """
        Args:
            table_uri: Delta table location (local path or cloud URI)
            partition_cols: Columns to partition by (default: ["data_version"])
            storage_options: Cloud storage credentials
        """
        pass
```

**Use Cases:**
- Production deployments
- Cloud storage (S3, GCS, Azure)
- Multi-user environments
- Data lakes

**Native Computation**: Potentially via DuckDB (can read Delta natively)

### DuckDB Backend

High-performance analytical database:

```python
class DuckDBMetadataStore(MetadataStore):
    """
    DuckDB storage backend.
    
    Features:
    - Embedded database (no server needed)
    - Native Delta Lake support
    - Extremely fast analytics
    - SQL-based data version computation
    - Excellent Parquet support
    """
    
    def __init__(
        self,
        db_path: str,
        *,
        read_only: bool = False,
        **kwargs
    ):
        """
        Args:
            db_path: Database file path (or :memory:)
            read_only: Open in read-only mode
        """
        pass
    
    def _compute_data_versions_native(self, ...):
        """Use native SQL with SHA256 for computation."""
        # Pure SQL implementation
        pass
```

**Use Cases:**
- Local development
- Single-machine production
- Fast prototyping
- Analytical queries

**Native Computation**: Full support via SQL + SHA256 functions

### ClickHouse Backend

Distributed analytical database:

```python
class ClickHouseMetadataStore(MetadataStore):
    """
    ClickHouse storage backend.
    
    Features:
    - Distributed computation
    - Column-oriented storage
    - Real-time analytics
    - SQL-based data version computation
    - Horizontal scalability
    """
    
    def __init__(
        self,
        connection_url: str,
        *,
        database: str = "metaxy",
        cluster: str | None = None,
        **kwargs
    ):
        """
        Args:
            connection_url: ClickHouse server URL
            database: Database name
            cluster: Cluster name for distributed tables
        """
        pass
    
    def _compute_data_versions_native(self, ...):
        """Use ClickHouse SQL with hash functions."""
        pass
```

**Use Cases:**
- Large-scale deployments
- Distributed systems
- Real-time dashboards
- Multi-tenant environments

**Native Computation**: Full support via SQL + hash functions

### SQLAlchemy Backend

Generic SQL database support:

```python
class SQLAlchemyMetadataStore(MetadataStore):
    """
    Generic SQL backend via SQLAlchemy.
    
    Supports: PostgreSQL, MySQL, SQLite, Oracle, etc.
    
    Features:
    - Wide database compatibility
    - Standard SQL interface
    - Transaction support
    - Optional native computation (DB-dependent)
    """
    
    def __init__(
        self,
        connection_url: str,
        *,
        table_prefix: str = "metaxy_",
        enable_native_computation: bool = False,
        **kwargs
    ):
        """
        Args:
            connection_url: SQLAlchemy connection string
            table_prefix: Prefix for table names
            enable_native_computation: Try to use DB-native SHA256
        """
        pass
```

**Use Cases:**
- Existing database infrastructure
- PostgreSQL deployments
- Traditional RDBMS environments
- Legacy system integration

**Native Computation**: Conditional (depends on database's SHA256 support)

## Advanced Partitioning

### Dynamic Partitioning

Automatically choose partition strategy based on data characteristics:

```python
class SmartPartitioningMixin:
    """
    Analyzes data to determine optimal partitioning strategy.
    
    Considers:
    - Data version cardinality
    - Sample count per version
    - Query patterns (if tracked)
    - Storage backend capabilities
    """
    
    def analyze_partition_strategy(
        self,
        feature: FeatureKey | type[Feature],
        sample_df: pl.DataFrame,
    ) -> list[str]:
        """Returns recommended partition columns."""
        pass
```

### Multi-Level Partitioning

Hierarchical partitioning for large-scale data:

```python
# Example: Partition by date, then by data_version
store = DeltaMetadataStore(
    table_uri="s3://bucket/metadata",
    partition_cols=["date", "data_version"],
)
```

### Time-Based Partitioning

For temporal data:

```python
# Example: Year/Month/Day partitioning
store = DeltaMetadataStore(
    table_uri="s3://bucket/metadata",
    partition_cols=["year", "month", "day"],
)
```

## Performance Optimizations

### Lazy Loading

Only load data when actually needed:

```python
class LazyDataFrame:
    """
    Lazy wrapper around metadata DataFrame.
    
    Defers actual data loading until operations are performed.
    Enables query optimization and predicate pushdown.
    """
    pass
```

### Query Pushdown

Push filters to storage layer:

```python
# Backend translates Polars expressions to native queries
df = store.read_metadata(
    MyFeature,
    filters=(pl.col("sample_id") > 1000) & (pl.col("region") == "us-west"),
)
# Executed as: SELECT * FROM feature WHERE sample_id > 1000 AND region = 'us-west'
```

### Incremental Materialization

Only materialize changed data:

```python
class IncrementalStore(MetadataStore):
    """
    Tracks which data has been materialized and only computes deltas.
    
    Useful for large features that change incrementally.
    """
    
    def materialize_incremental(
        self,
        feature: FeatureKey | type[Feature],
        sample_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Only compute data versions for new samples."""
        pass
```

### Caching

Cache frequently accessed metadata:

```python
class CachedMetadataStore(MetadataStore):
    """
    Wrapper that adds caching layer.
    
    Uses LRU cache for frequently accessed features.
    Safe due to immutability.
    """
    
    def __init__(
        self,
        store: MetadataStore,
        *,
        cache_size: int = 100,
        ttl: int | None = None,
    ):
        pass
```

## Distributed Computing

### Dask Integration

Process large datasets with Dask:

```python
import dask.dataframe as dd

class DaskMetadataStore(MetadataStore):
    """
    Distributed metadata processing via Dask.
    
    Features:
    - Parallel data version computation
    - Out-of-core processing
    - Distributed reads/writes
    """
    
    def calculate_and_write_data_versions(
        self,
        feature: FeatureKey | type[Feature],
        sample_df: dd.DataFrame,  # Dask DataFrame
        **kwargs
    ) -> dd.DataFrame:
        """Distributed computation."""
        pass
```

### Ray Integration

Distributed computing with Ray:

```python
class RayMetadataStore(MetadataStore):
    """
    Ray-based distributed metadata store.
    
    Features:
    - Actor-based data management
    - Distributed task execution
    - Fault tolerance
    """
    pass
```

### Spark Integration

For existing Spark infrastructure:

```python
class SparkMetadataStore(MetadataStore):
    """
    PySpark-based metadata store.
    
    Features:
    - Native Delta Lake support
    - Spark SQL computation
    - Existing cluster utilization
    """
    
    def calculate_and_write_data_versions(
        self,
        feature: FeatureKey | type[Feature],
        sample_df: DataFrame,  # Spark DataFrame
        **kwargs
    ) -> DataFrame:
        """Spark-based computation."""
        pass
```

## Monitoring and Observability

### Metrics Collection

Track store usage and performance:

```python
class InstrumentedMetadataStore(MetadataStore):
    """
    Wrapper that collects metrics.
    
    Tracks:
    - Read/write counts
    - Latencies
    - Data sizes
    - Error rates
    - Cache hit rates
    """
    
    def export_metrics(self) -> dict:
        """Export metrics for monitoring systems."""
        pass
```

### Logging

Structured logging for debugging:

```python
# Automatic logging of operations
store = DeltaMetadataStore(
    table_uri="s3://bucket/metadata",
    enable_logging=True,
    log_level="INFO",
)

# Logs operations like:
# [INFO] Writing metadata for feature=VideoProcessing, samples=1000, size=2.3MB
# [INFO] Reading metadata for feature=VideoProcessing, filters=applied, duration=45ms
```

## Testing Utilities

### Mock Stores

For testing without actual storage:

```python
class MockMetadataStore(MetadataStore):
    """
    Mock store for testing.
    
    Features:
    - Predictable behavior
    - Configurable failures
    - No I/O overhead
    """
    
    def set_failure(self, operation: str, exception: Exception):
        """Configure operation to fail."""
        pass
```

### Fixtures

Pytest fixtures for common testing scenarios:

```python
@pytest.fixture
def empty_store():
    """Empty in-memory store."""
    return InMemoryMetadataStore()

@pytest.fixture
def populated_store():
    """Store with sample data."""
    store = InMemoryMetadataStore()
    # ... populate with test data ...
    return store

@pytest.fixture
def multi_env_stores():
    """Dev/staging/prod store chain."""
    prod = InMemoryMetadataStore()
    staging = InMemoryMetadataStore(fallback_stores=[prod])
    dev = InMemoryMetadataStore(fallback_stores=[staging])
    return {"dev": dev, "staging": staging, "prod": prod}
```

## Data Migration

### Schema Evolution

Handle schema changes gracefully:

```python
class EvolvableMetadataStore(MetadataStore):
    """
    Supports schema evolution.
    
    Features:
    - Add new columns
    - Rename columns
    - Type conversions
    - Backward compatibility
    """
    
    def migrate_schema(
        self,
        feature: FeatureKey | type[Feature],
        migration: SchemaMigration,
    ) -> None:
        """Apply schema migration."""
        pass
```

### Data Version Migration

Recompute data versions with new code:

```python
def migrate_data_versions(
    store: MetadataStore,
    feature: FeatureKey | type[Feature],
    *,
    new_code_version: int,
) -> None:
    """
    Recompute all data versions with new code version.
    
    Useful when changing computation logic.
    """
    pass
```

## Security and Access Control

### Encryption

Encrypt sensitive metadata:

```python
class EncryptedMetadataStore(MetadataStore):
    """
    Transparent encryption wrapper.
    
    Features:
    - Column-level encryption
    - Key rotation
    - Multiple encryption algorithms
    """
    
    def __init__(
        self,
        store: MetadataStore,
        *,
        encryption_key: bytes,
        encrypted_columns: list[str],
    ):
        pass
```

### Access Control

Role-based access control:

```python
class SecureMetadataStore(MetadataStore):
    """
    RBAC wrapper for metadata store.
    
    Features:
    - User authentication
    - Permission management
    - Audit logging
    """
    
    def __init__(
        self,
        store: MetadataStore,
        *,
        auth_backend: AuthBackend,
    ):
        pass
```

## Priority Order

### Short Term (3-6 months)
1. MaterializationCatalog (SQLite implementation)
2. Delta Lake backend
3. DuckDB backend with native computation
4. Basic partitioning support

### Medium Term (6-12 months)
1. ClickHouse backend
2. SQLAlchemy backend
3. Caching layer
4. Query optimization

### Long Term (12+ months)
1. Distributed computing integration (Dask/Ray/Spark)
2. Advanced monitoring
3. Schema evolution
4. Security features

## Contributing

We welcome contributions! Priority areas:
- Additional storage backends
- Performance optimizations
- Documentation improvements
- Test coverage
- Real-world usage examples

See CONTRIBUTING.md for guidelines (coming soon).
