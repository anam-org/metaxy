# Metaxy Performance Benchmarks

Performance benchmarks for ADBC metadata stores vs Ibis-based stores.

## Overview

These benchmarks validate the performance claims in the ADBC documentation:

- 2-6x faster single-threaded writes
- 5-10x faster concurrent writes (with connection pooling)

## Prerequisites

### DuckDB and SQLite Benchmarks

No external dependencies required. These benchmarks use temporary in-memory/file databases.

### PostgreSQL Benchmarks

Requires a running PostgreSQL server:

```bash
# Start PostgreSQL via Docker
docker run -d \
  --name metaxy-bench-postgres \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  postgres:17

# Stop when done
docker stop metaxy-bench-postgres
docker rm metaxy-bench-postgres
```

## Running Benchmarks

### Quick Start (DuckDB)

```bash
# Benchmark DuckDB with 10k rows
uv run python benchmarks/adbc_vs_ibis.py --backend duckdb --rows 10000

# Benchmark DuckDB with 100k rows
uv run python benchmarks/adbc_vs_ibis.py --backend duckdb --rows 100000
```

### All Backends

```bash
# Benchmark all backends (requires PostgreSQL running)
uv run python benchmarks/adbc_vs_ibis.py --backend all --rows 50000
```

### SQLite Only

```bash
uv run python benchmarks/adbc_vs_ibis.py --backend sqlite --rows 10000
```

### PostgreSQL Only

```bash
# Make sure PostgreSQL is running first
uv run python benchmarks/adbc_vs_ibis.py --backend postgres --rows 100000

# Custom PostgreSQL connection
uv run python benchmarks/adbc_vs_ibis.py \
  --backend postgres \
  --rows 50000 \
  --postgres-url "postgresql://user:pass@host:5432/db"
```

## Interpreting Results

Example output:

```
================================================================================
ADBC vs Ibis Write Performance Benchmark
================================================================================

DUCKDB (10,000 rows)
--------------------------------------------------------------------------------
  Ibis:  1.234s  (8,103 rows/sec)
  ADBC:  0.456s  (21,930 rows/sec)
  Speedup: 2.71x
  ✓ ADBC is 2x+ faster

POSTGRES (10,000 rows)
--------------------------------------------------------------------------------
  Ibis:  2.456s  (4,072 rows/sec)
  ADBC:  0.987s  (10,132 rows/sec)
  Speedup: 2.49x
  ✓ ADBC is 2x+ faster
```

### Success Criteria

- **Speedup ≥ 2.0x**: ✓ ADBC is 2x+ faster (target met)
- **Speedup 1.5-2.0x**: ⚠ ADBC is faster but <2x target (investigate)
- **Speedup < 1.5x**: ✗ ADBC not significantly faster (issue)

## Factors Affecting Performance

### Data Size

Larger datasets show more pronounced speedups due to:

- Amortized connection overhead
- Bulk transfer efficiency
- Arrow zero-copy benefits

### Hardware

- **CPU**: Single-threaded benchmarks are CPU-bound
- **Disk I/O**: File-based databases (DuckDB, SQLite) affected by disk speed
- **Network**: PostgreSQL affected by network latency

### Database Configuration

PostgreSQL performance depends on:

- `shared_buffers` setting
- `work_mem` allocation
- Connection pooling configuration
- Running locally vs remotely

## Benchmark Implementation

### Write Operation

Each benchmark:

1. Generates synthetic data (sample_id, value, score)
2. Drops existing table (if present)
3. Times `write_metadata()` call
4. Calculates rows/second throughput

### Fairness

- Same data for Ibis and ADBC
- Same hash algorithm (MD5 or XXHASH64)
- Single connection (max_connections=1) for ADBC
- Fresh database for each run

## Known Limitations

### Current Implementation

The `write_metadata_bulk()` API currently delegates to `write_metadata()`, so these benchmarks don't yet measure the full concurrent write performance. Future benchmarks will test:

- Concurrent writes with connection pooling
- DataFrame partitioning efficiency
- RecordBatch streaming overhead

### Measurement Precision

- Times include hash computation, not just ADBC operations
- First run may be slower due to cold cache
- Run multiple times for stable measurements

## Future Benchmarks

Planned additions:

- Bulk ingestion with concurrency=8 (target: 5-10x speedup)
- Read performance comparison
- Federation query performance
- Flight SQL roundtrip latency
- Memory usage profiling

## Troubleshooting

### PostgreSQL Connection Errors

```
⚠ PostgreSQL benchmark failed: connection refused
```

**Solution**: Ensure PostgreSQL is running and accessible at the specified URL.

### Permission Errors

```
PermissionError: [Errno 13] Permission denied
```

**Solution**: Check file permissions for temporary directory or database files.

### Import Errors

```
ModuleNotFoundError: No module named 'metaxy.metadata_store.adbc_postgres'
```

**Solution**: Install ADBC dependencies:

```bash
uv sync --extra adbc
```

## Contributing

To add new benchmarks:

1. Create benchmark script in `benchmarks/`
2. Document in this README
3. Follow existing naming conventions
4. Include usage examples
5. Add to CI pipeline (future work)
