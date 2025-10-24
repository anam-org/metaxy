# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Always use the following agents:
- @agent-planner
- @agent-python-dev
- @agent-qa
- @agent-python-test-engineer

## Project Status

**⚠️ Early Development - No Stable API**

This project is in active development with no users yet. Breaking changes are expected and encouraged:
- **No backward compatibility required** - **we have no users yet**, refactor freely, prioritize changes that improve design and functionality
- **API changes allowed** - improve interfaces without deprecation warnings
- **Breaking changes welcome** - prioritize better design over stability
- **Move fast** - optimize for the best long-term architecture, not short-term compatibility

## Project Overview

Metaxy is a feature metadata management system for multimodal ML pipelines that tracks feature versions, dependencies, and data lineage. It enables:

- **Declarative feature definitions** with explicit dependencies and versioning
- **Automatic change detection** to identify affected downstream features
- **Smart migrations** to reconcile metadata when code refactors don't change computation
- **Dependency-aware updates** with automatic recomputation when upstream dependencies change
- **Immutable metadata** with copy-on-write storage preserving historical versions
- **Graph snapshots** recording complete feature graph states in deployment pipelines

## Development

Never create git commits unless asked for explicitly by the user.

### Environment Setup
```bash
# Install dependencies (uv is required)
uv sync

# Install with optional dependencies
uv sync --extra ibis
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_migrations.py

# Run specific test
uv run pytest tests/test_migrations.py::test_migration_generation

# Run tests with specific backend
uv run pytest tests/metadata_stores/test_duckdb.py
```

Always keep tests up-to-date and maintainable. Add or update tests as features are added or modified.

### Linting and Formatting
```bash
# Run ruff linter
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Type Checking
```bash
# Run pyrefly type checker
uv run basedpyright --level error
```

### CLI Usage
```bash
# Run metaxy CLI (during development)
uv run metaxy --help

# List features
uv run metaxy list features

# Push graph snapshot (CD workflow)
uv run metaxy graph push

# Generate migration
uv run metaxy migrations generate

# Apply migrations
uv run metaxy migrations apply
```

### Examples
```bash
# Run examples - each example is a directory with metaxy.toml
cd examples/src/<example_name>  # e.g., recompute, migration

# Set VERSION to switch between feature definition variants
VERSION=1 uv run metaxy list features
VERSION=2 uv run metaxy list features

# See examples/README.md for details
```

## Architecture

### Core Components

#### 1. Feature Graph (`src/metaxy/models/feature.py`)
Central registry managing feature definitions and their relationships:
- **FeatureGraph**: Tracks all features by key, computes feature versions, and manages graph snapshots
- **Feature (base class)**: All features inherit from this with `spec=FeatureSpec(...)` parameter
- **Active graph context**: Uses context variables (`_active_graph`) to support multiple graphs in testing/migrations
- **Snapshot version**: Deterministic hash of entire graph state (all feature versions) for deployment tracking

Key methods:
- `FeatureGraph.get_active()`: Returns currently active graph (default or from context)
- `FeatureGraph.from_snapshot()`: Reconstructs graph from DB snapshot by importing Feature classes (used in migrations)
- `Feature.feature_version()`: Returns hash of feature definition (deps + fields + code_versions)

#### 2. Metadata Store (`src/metaxy/metadata_store/base.py`)
Abstract base class for metadata storage backends:
- **Immutable storage**: Append-only writes with copy-on-write semantics
- **Fallback store chains**: Composable read-through cache for branch deployments
- **Three-component architecture**: UpstreamJoiner, DataVersionCalculator, MetadataDiffResolver
- **Backend-agnostic**: Generic type `TRef` allows different backend table references
- **Narwhals interface**: Public API uses Narwhals DataFrames/LazyFrames for cross-backend compatibility
- **Native vs Polars components**: Stores choose components based on capabilities:
  - **native data version calculations** (e.g., Ibis-based for DuckDB/ClickHouse): Execute all operations (joins, hashing, diffs) directly in the database, only pulling out final results. This minimizes data transfer and leverages database query optimization.
  - **Polars components**: Pull data into memory when fallback stores are used or store lacks compute support (InMemory, SQLite, DeltaLake)

Implementations:
- `InMemoryMetadataStore` (memory.py): Polars DataFrames in memory
- `IbisMetadataStore` (ibis.py): Abstract for SQL databases
- `DuckDBMetadataStore` (duckdb.py): DuckDB backend
- `SQLiteMetadataStore` (sqlite.py): SQLite backend
- `ClickHouseMetadataStore` (clickhouse.py): ClickHouse backend

Key system tables (stored with prefix `metaxy-system`):
- `feature_versions`: Tracks when each feature version was recorded (populated by `metaxy graph push`)
- `migrations`: Tracks applied migrations and their status

#### 3. Data Versioning (`src/metaxy/data_versioning/`)
Three-component architecture for calculating and comparing data versions:

**UpstreamJoiner** (`joiners/`): Joins upstream feature metadata
- `NarwhalsJoiner`: Primary implementation using Narwhals for backend-agnostic joins
- Default: Inner join on `sample_uid`
- Native implementations execute joins directly in the database

**DataVersionCalculator** (`calculators/`): Computes data version hashes
- `PolarsDataVersionCalculator`: Uses `polars_hash` plugin for fast hashing (used when data is in memory)
- Native implementations (e.g., Ibis-based): Execute hash calculations directly in the database using SQL/native functions
- Supports multiple hash algorithms (xxhash, sha256, etc.)
- Creates nested struct column: `data_version: {field1: hash, field2: hash}`
- **Native approach**: All computations stay in the database, only final data versions are pulled out
- **Polars approach**: Used when fallback stores are needed, or when the store doesn't support native compute (InMemory, SQLite, DeltaLake)

**MetadataDiffResolver** (`diff/`): Compares target vs current versions
- `NarwhalsDiffResolver`: Primary backend-agnostic comparison using Narwhals
- Native implementations execute diffs (anti-joins, comparisons) directly in the database
- Returns: `DiffResult(added, changed, removed)` or `LazyDiffResult` (lazy frames with Narwhals)
- Only pulls necessary data (samples that need updating) out of the database

#### 4. Migration System (`src/metaxy/migrations/`)
Handles metadata updates when feature definitions change:
- **Migration detection**: Compares latest snapshot in store vs current code
- **Explicit operations**: All affected features (root + downstream) listed in YAML
- **Idempotent execution**: Safely re-runnable, recovers from partial failures
- **DataVersionReconciliation**: Operation type for code refactors that don't change computation
- **Snapshot-based**: References `from_snapshot_version` and `to_snapshot_version` to derive feature versions
- **Requires Feature classes**: Imports actual Feature classes (via `FeatureGraph.from_snapshot()`) to support custom `load_input()` methods

Migration workflow:
1. `metaxy graph push` in CD to record feature graph snapshot
2. `metaxy migrations generate` to detect changes and create YAML
3. Review migration YAML (check reasons, validate operations)
4. `metaxy migrations apply` to execute operations (imports Feature classes for custom alignment logic)

#### 5. CLI (`src/metaxy/cli/`)
Command-line interface built with cyclopts:
- `app.py`: Main entry point and command routing
- `context.py`: Manages configuration and context
- `migrations.py`: Migration commands (generate, scaffold, apply, status)
- `push.py`: Graph snapshot recording
- `list.py`: List features and entities

### Key Design Patterns

#### Immutable Metadata
All metadata writes are append-only. When migrations update metadata:
1. Query rows with old `feature_version`
2. Copy all user columns (preserving custom metadata)
3. Recalculate `data_version` based on new feature definition
4. Write new rows with new `feature_version` and `snapshot_version`
5. Old rows remain for historical queries and audit trail

#### Feature Version vs Data Version
- **Feature version**: Hash of feature definition (code, deps, fields). Deterministic from code alone.
- **Data version**: Hash of upstream data versions for a specific sample. Depends on actual data.
- **Snapshot version**: Hash of all feature versions in graph. Represents entire graph state.

Metadata rows have:
```python
{
    "sample_uid": 123,
    "data_version": {"field1": "hash1", "field2": "hash2"},  # Struct column
    "feature_version": "abc123",  # From feature definition
    "snapshot_version": "def456",      # From graph snapshot
    ...user columns...
}
```

#### Graph Context Management
The active graph is managed via context variables:
```python
# Default global graph (used by imports at module level)
graph = FeatureGraph()

# Get active graph
active = FeatureGraph.get_active()

# Use custom graph temporarily
with custom_graph.use():
    # All operations use custom_graph
    pass
```

This enables:
- Testing with isolated feature registries
- Migration operations with historical graphs
- Multi-graph applications

#### Custom Metadata Alignment
Features can override `load_input()` for custom join logic:
- **Default**: Inner join on `sample_uid` (only samples in ALL upstream features)
- **One-to-many**: Generate multiple child samples per parent (e.g., video frames)
- **Filtering**: Only process samples meeting certain conditions
- **Outer join**: Keep union of all upstream samples

This is critical for migrations when upstream dependencies change.

## Important Constraints

### Narwhals as the Public Interface
**Important**: The codebase uses Narwhals as the primary user-facing API:
- All public methods accept and return `nw.DataFrame[Any]` or `nw.LazyFrame[Any]`
- When writing code, prefer Narwhals operations over backend-specific code
- The `to_native()` method converts Narwhals to the underlying backend type when needed

**Native vs Polars Components**:
The store automatically selects the optimal component strategy:

**native data version calculations** (preferred when available):
- Execute all operations (joins, hashing, diffs) directly in the database
- Only pull final results into memory (e.g., list of samples that need updating)
- Leverage database query optimization and avoid data transfer overhead
- Used by: DuckDB, ClickHouse, and other SQL databases with compute capabilities

**Polars components** (fallback):
Used in specific cases:
1. **Fallback store scenarios**: When upstream metadata needs to be pulled from fallback stores (cross-store operations require in-memory processing)
2. **Non-compute stores**: Stores without native compute/hashing support (InMemory, SQLite, DeltaLake)
3. **User preference**: Can be forced via `prefer_native=False` parameter

### Module-Level Import Restrictions
From `pyproject.toml`, these modules are banned at module level (must be lazy-imported in functions):
```python
# ❌ Don't do this
import ibis
import duckdb

# ✅ Do this instead
def my_function():
    import ibis
    import duckdb
```

This prevents unnecessary dependencies from loading when not needed.

### imports

# ❌ Don't do this
from __future__ import annotations
this is not needed for modern versions of python!

### Hash Algorithm Consistency
All stores in a fallback chain must use the same hash algorithm. This is validated at store open time.

### Migration Prerequisites
The migration system requires `metaxy graph push` to be run in CD workflows. Without recorded snapshots, migration detection cannot work (no baseline to compare against).

## Testing Patterns

### Feature Graph Isolation
Always use isolated graphs in tests:
```python
def test_my_feature():
    test_graph = FeatureGraph()
    with test_graph.use():
        class MyFeature(Feature, spec=...):
            pass
        # Test operations here
```

### Metadata Store Context Managers
Stores must be used as context managers:
```python
with InMemoryMetadataStore() as store:
    store.write_metadata(MyFeature, df)
```

### Snapshot Testing
Uses `syrupy` for snapshot testing. Snapshots stored in `__snapshots__/` directories.

## Common Workflows

### Adding a New Metadata Store Backend
1. Inherit from `MetadataStore[TRef]` with appropriate `TRef` type
2. Implement abstract methods: `_get_default_hash_algorithm()`, `_supports_native_components()`, `_create_native_components()`, `open()`, `close()`, `_write_metadata_impl()`, `_read_metadata_native()`, `_drop_feature_metadata_impl()`, `_list_features_local()`
3. Implement reference conversion methods: `_feature_to_ref()`, `_sample_to_ref()`, `_result_to_dataframe()`, `_dataframe_to_ref()`
4. Add tests in `tests/metadata_stores/`

### Adding New Hash Algorithm Support
1. Add to `HashAlgorithm` enum in `data_versioning/hash_algorithms.py`
2. Update `PolarsDataVersionCalculator.supported_algorithms`
3. Add hash expression in calculator's `_get_hash_expr()` method
4. Add tests for new algorithm

### Extending Migration Operations
1. Create operation class in `migrations/ops.py` inheriting from `BaseOperation`
2. Implement `execute()` method
3. Update migration YAML schema to support new operation type
4. Add tests in `tests/migrations`

## Key Files Reference

### Core Models
- `models/feature.py`: Feature, FeatureGraph, graph context management
- `models/feature_spec.py`: FeatureSpec, FieldSpec, dependency specifications
- `models/types.py`: FeatureKey, FieldKey, type definitions
- `models/graph.py`: FeatureGraph, dependency resolution

### Metadata Storage
- `metadata_store/base.py`: Abstract MetadataStore with full API
- `metadata_store/memory.py`: In-memory implementation (simplest to understand)
- `metadata_store/ibis.py`: SQL backend base class
- `metadata_store/exceptions.py`: All store-related exceptions

### Data Versioning
- `data_versioning/calculators/base.py`: DataVersionCalculator interface
- `data_versioning/joiners/base.py`: UpstreamJoiner interface
- `data_versioning/diff/base.py`: MetadataDiffResolver interface
- `data_versioning/hash_algorithms.py`: HashAlgorithm enum

### Migrations
- `migrations/detector.py`: Change detection logic
- `migrations/ops.py`: Migration operation types
- `migrations/models.py`: Migration YAML schema (Pydantic models)

### CLI
- `cli/app.py`: Main CLI entry point
- `cli/migrations.py`: Migration commands implementation
