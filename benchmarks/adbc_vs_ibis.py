"""Benchmark ADBC vs Ibis metadata store write performance.

This script compares write performance between ADBC and Ibis-based stores
to validate the performance claims in the documentation.

Usage:
    python benchmarks/adbc_vs_ibis.py --backend duckdb --rows 10000
    python benchmarks/adbc_vs_ibis.py --backend postgres --rows 100000
    python benchmarks/adbc_vs_ibis.py --backend all

Requirements:
    - DuckDB: No external dependencies
    - SQLite: No external dependencies
    - PostgreSQL: Requires running PostgreSQL server
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import polars as pl

import metaxy as mx
from metaxy import HashAlgorithm
from metaxy._testing import add_metaxy_provenance_column


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    backend: str
    store_type: str  # "ibis" or "adbc"
    rows: int
    write_time: float
    rows_per_sec: float


class SimpleBenchFeature(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="bench",
        id_columns=["sample_id"],
        fields=["value", "score"],
    ),
):
    """Simple feature for benchmarking."""

    sample_id: int
    value: str
    score: float


def generate_test_data(n_rows: int) -> pl.DataFrame:
    """Generate test data for benchmarking.

    Args:
        n_rows: Number of rows to generate

    Returns:
        Polars DataFrame with test data including provenance
    """
    return pl.DataFrame(
        {
            "sample_id": range(n_rows),
            "value": [f"value_{i}" for i in range(n_rows)],
            "score": [float(i) * 0.1 for i in range(n_rows)],
            "metaxy_provenance_by_field": [{"value": f"hash_{i}", "score": f"hash_{i}"} for i in range(n_rows)],
        }
    )


@contextmanager
def temp_duckdb():
    """Create temporary DuckDB database (in-memory)."""
    # Use in-memory database to avoid file system issues
    yield ":memory:"


@contextmanager
def temp_sqlite():
    """Create temporary SQLite database (in-memory)."""
    # Use in-memory database to avoid file system issues
    yield ":memory:"


def benchmark_write(store: Any, feature: type[mx.BaseFeature], df: pl.DataFrame) -> float:
    """Benchmark write operation.

    Args:
        store: Metadata store instance
        feature: Feature class
        df: Data to write (without provenance columns)

    Returns:
        Time taken in seconds
    """
    # Add provenance columns
    df_with_provenance = add_metaxy_provenance_column(df, feature)

    with store.open("write"):
        # Drop table if exists
        try:
            store.drop_metadata(feature)
        except Exception:
            pass

        # Time the write operation
        start = time.perf_counter()
        store.write_metadata(feature, df_with_provenance)
        end = time.perf_counter()

    return end - start


def benchmark_duckdb(n_rows: int) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Benchmark DuckDB ADBC vs Ibis.

    Args:
        n_rows: Number of rows to write

    Returns:
        Tuple of (ibis_result, adbc_result)
    """
    from metaxy.metadata_store.adbc_duckdb import ADBCDuckDBMetadataStore
    from metaxy.metadata_store.duckdb import DuckDBMetadataStore

    df = generate_test_data(n_rows)

    # Benchmark Ibis
    with temp_duckdb() as db_path:
        store_ibis = DuckDBMetadataStore(
            database=db_path,
            hash_algorithm=HashAlgorithm.XXHASH64,
            auto_create_tables=True,
        )
        time_ibis = benchmark_write(store_ibis, SimpleBenchFeature, df)

    # Benchmark ADBC
    with temp_duckdb() as db_path:
        store_adbc = ADBCDuckDBMetadataStore(
            database=db_path,
            hash_algorithm=HashAlgorithm.XXHASH64,
            max_connections=1,
            auto_create_tables=True,
        )
        time_adbc = benchmark_write(store_adbc, SimpleBenchFeature, df)

    return (
        BenchmarkResult("duckdb", "ibis", n_rows, time_ibis, n_rows / time_ibis),
        BenchmarkResult("duckdb", "adbc", n_rows, time_adbc, n_rows / time_adbc),
    )


def benchmark_sqlite(n_rows: int) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Benchmark SQLite ADBC (no Ibis comparison - SQLite has no Ibis store).

    Args:
        n_rows: Number of rows to write

    Returns:
        Tuple of placeholder results (ADBC only)

    Note:
        SQLite doesn't have an Ibis-based metadata store, so we can't compare.
        This function is kept for API consistency but returns dummy results.
    """
    from metaxy.metadata_store.adbc_sqlite import ADBCSQLiteMetadataStore

    df = generate_test_data(n_rows)

    # Benchmark ADBC only
    with temp_sqlite() as db_path:
        store_adbc = ADBCSQLiteMetadataStore(
            database=db_path,
            hash_algorithm=HashAlgorithm.MD5,
            max_connections=1,
            auto_create_tables=True,
        )
        time_adbc = benchmark_write(store_adbc, SimpleBenchFeature, df)

    # Return dummy Ibis result and real ADBC result
    # (SQLite has no Ibis store to compare against)
    return (
        BenchmarkResult("sqlite", "ibis", n_rows, float("inf"), 0.0),  # Placeholder
        BenchmarkResult("sqlite", "adbc", n_rows, time_adbc, n_rows / time_adbc),
    )


def benchmark_postgres(n_rows: int, connection_string: str) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Benchmark PostgreSQL ADBC vs Ibis.

    Args:
        n_rows: Number of rows to write
        connection_string: PostgreSQL connection string

    Returns:
        Tuple of (ibis_result, adbc_result)
    """
    from metaxy.metadata_store.postgres import PostgresMetadataStore

    from metaxy.metadata_store.adbc_postgres import ADBCPostgresMetadataStore

    df = generate_test_data(n_rows)

    # Benchmark Ibis
    store_ibis = PostgresMetadataStore(
        connection_string=connection_string,
        hash_algorithm=HashAlgorithm.MD5,
        auto_create_tables=True,
    )
    time_ibis = benchmark_write(store_ibis, SimpleBenchFeature, df)

    # Benchmark ADBC
    store_adbc = ADBCPostgresMetadataStore(
        connection_string=connection_string,
        hash_algorithm=HashAlgorithm.MD5,
        max_connections=1,
        auto_create_tables=True,
    )
    time_adbc = benchmark_write(store_adbc, SimpleBenchFeature, df)

    return (
        BenchmarkResult("postgres", "ibis", n_rows, time_ibis, n_rows / time_ibis),
        BenchmarkResult("postgres", "adbc", n_rows, time_adbc, n_rows / time_adbc),
    )


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table.

    Args:
        results: List of benchmark results
    """
    print("\n" + "=" * 80)
    print("ADBC vs Ibis Write Performance Benchmark")
    print("=" * 80)

    # Group by backend
    backends = {r.backend for r in results}

    for backend in sorted(backends):
        backend_results = [r for r in results if r.backend == backend]
        ibis_result = next(r for r in backend_results if r.store_type == "ibis")
        adbc_result = next(r for r in backend_results if r.store_type == "adbc")

        # Skip backends without Ibis implementation
        if ibis_result.write_time == float("inf"):
            print(f"\n{backend.upper()} ({adbc_result.rows:,} rows)")
            print("-" * 80)
            print(f"  ADBC:  {adbc_result.write_time:.3f}s  ({adbc_result.rows_per_sec:,.0f} rows/sec)")
            print("  (No Ibis implementation to compare against)")
            continue

        speedup = adbc_result.rows_per_sec / ibis_result.rows_per_sec

        print(f"\n{backend.upper()} ({ibis_result.rows:,} rows)")
        print("-" * 80)
        print(f"  Ibis:  {ibis_result.write_time:.3f}s  ({ibis_result.rows_per_sec:,.0f} rows/sec)")
        print(f"  ADBC:  {adbc_result.write_time:.3f}s  ({adbc_result.rows_per_sec:,.0f} rows/sec)")
        print(f"  Speedup: {speedup:.2f}x")

        if speedup >= 2.0:
            print("  ✓ ADBC is 2x+ faster")
        elif speedup >= 1.5:
            print("  ⚠ ADBC is faster but <2x target")
        else:
            print("  ✗ ADBC not significantly faster")

    print("\n" + "=" * 80)


def main():
    """Run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark ADBC vs Ibis write performance")
    parser.add_argument(
        "--backend",
        choices=["duckdb", "sqlite", "postgres", "all"],
        default="duckdb",
        help="Which backend to benchmark",
    )
    parser.add_argument("--rows", type=int, default=10000, help="Number of rows to write")
    parser.add_argument(
        "--postgres-url",
        default="postgresql://postgres:postgres@localhost:5432/postgres",
        help="PostgreSQL connection string",
    )
    args = parser.parse_args()

    results: list[BenchmarkResult] = []

    if args.backend in ("duckdb", "all"):
        print(f"Benchmarking DuckDB with {args.rows:,} rows...")
        ibis, adbc = benchmark_duckdb(args.rows)
        results.extend([ibis, adbc])

    if args.backend in ("sqlite", "all"):
        print(f"Benchmarking SQLite with {args.rows:,} rows...")
        ibis, adbc = benchmark_sqlite(args.rows)
        results.extend([ibis, adbc])

    if args.backend in ("postgres", "all"):
        print(f"Benchmarking PostgreSQL with {args.rows:,} rows...")
        try:
            ibis, adbc = benchmark_postgres(args.rows, args.postgres_url)
            results.extend([ibis, adbc])
        except Exception as e:
            print(f"⚠ PostgreSQL benchmark failed: {e}")
            print("  Make sure PostgreSQL is running and accessible.")

    print_results(results)


if __name__ == "__main__":
    main()
