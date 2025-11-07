# Python Migration Examples

This directory contains comprehensive examples demonstrating different Python migration patterns in Metaxy. These examples show how to use the Python migration API to handle complex migration scenarios beyond what YAML migrations can express.

> **Note on Type Checking:** These examples intentionally demonstrate advanced patterns that may produce type checker warnings (e.g., optional dependencies such as `boto3`/`psycopg2`, dynamic imports for tests). These patterns are educational and show how to handle dynamic operations and external integrations. In production code, you may choose different approaches based on your requirements.

## Overview

Metaxy supports two migration formats:

1. **YAML Migrations** (`.yaml`): Simple, declarative migrations for common cases
2. **Python Migrations** (`.py`): Programmatic migrations for complex logic, validation, and custom operations

Python migrations provide full control over migration behavior, enabling:

- Custom execution logic
- Dynamic operation generation
- Pydantic validation
- External resource integration (S3, databases, APIs)
- Reusable operation classes

## When to Use Python vs YAML

### Use YAML Migrations When:

- Simple `DataVersionReconciliation` is sufficient
- Operations are known at migration creation time
- No custom logic or validation needed
- Standard workflow is adequate

### Use Python Migrations When:

- Need custom execution logic
- Operations computed dynamically at runtime
- Validation of configuration or external resources required
- Loading data from external sources (S3, databases, APIs)
- Building reusable operation classes
- Complex conditional logic based on feature inspection
- Custom error handling and retry logic

## Quick Start

### `example_python_migration.py`

A simple reference file showing the basic structure of Python migrations. Contains two example classes:

- `ExampleMigration`: Minimal PythonMigration showing required fields
- `CustomBackfillMigration`: Example of overriding `execute()` for custom logic

> **Note:** This file contains multiple Migration classes for reference purposes and is not meant to be loaded directly (each migration file should contain exactly one Migration class).

For complete, runnable examples, see the numbered examples below.

## Examples

### Example 1: Simple PythonMigration Reconciliation

**File:** `20250101_120000_simple_reconciliation.py`

The simplest Python migration demonstrating the `PythonMigration` base class with `DataVersionReconciliation`.

**Use Case:** Code refactoring that doesn't change computation results.

**Key Concepts:**

- `PythonMigration` class definition
- `build_operations()` returning strongly typed operations
- Snapshot version references
- Parent migration chain

**When to Use:**

- Dependency graph updates
- Field schema improvements
- Type annotation changes
- Import reorganization

```python
class SimpleReconciliationMigration(PythonMigration):
    migration_id: str = "20250101_120000_simple_reconciliation"
    created_at: datetime = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    parent: str = "initial"
    from_snapshot_version: str = "abc123..."
    to_snapshot_version: str = "def456..."

    def build_operations(self) -> list[DataVersionReconciliation]:
        return [
            DataVersionReconciliation(),
        ]
```

---

### Example 2: CustomMigration with S3 Backfill

**File:** `20250101_130000_custom_backfill.py`

Custom migration that loads video metadata from S3, filters by size, and writes to the metadata store.

**Use Case:** Backfilling metadata from external sources.

**Key Concepts:**

- CustomMigration class
- Full control over execute() logic
- Loading from external sources (S3)
- Custom filtering and transformation
- Joining with Metaxy field_provenance
- Error handling and progress tracking

**When to Use:**

- Migrating from legacy systems
- Importing historical data
- Integrating with external services
- Bulk loading from data lakes

```python
class S3VideoBackfillMigration(CustomMigration):
    s3_bucket: str = "prod-videos"
    s3_prefix: str = "processed/"
    min_size_mb: int = 10

    def execute(self, store, project, *, dry_run=False):
        # 1. Load from S3
        # 2. Filter by criteria
        # 3. Get field_provenance from Metaxy
        # 4. Join and write to store
        ...
```

---

### Example 3: PythonMigration with Dynamic Operations

**File:** `20250101_140000_dynamic_operations.py`

PythonMigration where operations are computed at runtime based on feature inspection, configuration, or environmental conditions.

**Use Case:** Migrations with conditional operation logic.

**Key Concepts:**

- `build_operations()` for dynamic ops
- Runtime feature graph inspection
- Configuration-driven operation selection
- Environment-specific behavior
- Filtering affected features

**When to Use:**

- Operations depend on runtime conditions
- Different strategies for different environments
- Configuration from external sources
- A/B testing migration strategies
- Conditional logic for operation selection

```python
class DynamicOperationsMigration(PythonMigration):
    enable_custom_validation: bool = True
    skip_root_features: bool = False

    def build_operations(self) -> list[dict[str, str]]:
        # Compute operations dynamically
        graph = FeatureGraph.get_active()
        operations = []

        # Add operations based on conditions
        if self.enable_custom_validation:
            operations.append({"type": "..."})

        return operations
```

---

### Example 4: CustomMigration with Validation

**File:** `20250101_150000_custom_validation.py`

Custom migration demonstrating comprehensive Pydantic validation for configuration, external resources, and data quality.

**Use Case:** Ensuring migration configuration is valid before execution.

**Key Concepts:**

- `@field_validator` for individual field validation
- `@model_validator` for cross-field validation
- Database connectivity validation
- File path and permission checks
- Date range validation
- Numeric range constraints
- Fail-fast validation

**When to Use:**

- Complex configuration needs validation
- External dependencies should be checked upfront
- Data quality requirements
- Environment prerequisite checks
- Want clear error messages before expensive operations

```python
class ValidatedDatabaseBackfillMigration(CustomMigration):
    database_url: str
    table_name: str
    batch_size: int = 1000

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith(("postgresql://", "postgres://")):
            raise ValueError("Invalid database URL")
        return v

    @model_validator(mode="after")
    def validate_database_connectivity(self):
        # Check database is accessible
        ...
        return self
```

---

### Example 5: MetadataBackfill Subclass

**File:** `20250101_160000_metadata_backfill.py`

Demonstrates creating reusable `MetadataBackfill` operation classes that can be used in multiple migrations.

**Use Case:** Building a library of reusable backfill operations.

**Key Concepts:**

- MetadataBackfill base class
- Reusable operation design
- Composition with DiffMigration
- Operation contract (execute returns row count)
- S3, database, and API backfill patterns
- Batch processing
- Pydantic validation in operations

**When to Use:**

- Reusable backfill logic for specific data sources
- Want operation-level abstraction
- Building a library of backfill operations
- Need to combine with DataVersionReconciliation
- Use in multiple migrations

```python
class S3VideoBackfillOperation(MetadataBackfill):
    type: Literal["metaxy.migrations.ops.S3VideoBackfillOperation"]
    s3_bucket: str
    s3_prefix: str
    min_size_mb: int = 1

    def execute(
        self, store, *, from_snapshot_version, to_snapshot_version, dry_run=False
    ):
        # 1. List S3 objects
        # 2. Filter and transform
        # 3. Get field_provenance
        # 4. Join and write
        return rows_written
```

**Usage in DiffMigration:**

```yaml
ops:
  - type: metaxy.migrations.ops.S3VideoBackfillOperation
    id: backfill_videos
    feature_key: ["video", "files"]
    s3_bucket: prod-videos
    s3_prefix: processed/
    reason: Initial backfill
```

## Migration Class Hierarchy

```
Migration (abstract base)
├── DiffMigration
│   └── Uses snapshot diff to compute affected features
│   └── Supports DataVersionReconciliation operation
│   └── Can have dynamic operations via build_operations()
│
├── CustomMigration
│   └── Full control over execute() logic
│   └── No automatic affected feature computation
│   └── Complete flexibility
│
└── FullGraphMigration
    └── Operates within single snapshot
    └── No snapshot diff needed
```

## Operation Class Hierarchy

```
BaseOperation (abstract)
└── Used in DiffMigration ops list
└── Must implement execute()

MetadataBackfill (extends BaseOperation)
└── Structured interface for backfills
└── Reusable across migrations
└── Can be used in YAML or Python

DataVersionReconciliation
└── Special operation for reconciliation
└── No user configuration needed
└── Applies to all affected features
```

## Quick Reference

### DiffMigration vs CustomMigration vs MetadataBackfill

| Feature                     | DiffMigration    | CustomMigration | MetadataBackfill  |
| --------------------------- | ---------------- | --------------- | ----------------- |
| Automatic affected features | Yes              | No              | N/A (operation)   |
| Snapshot diff required      | Yes              | No              | No                |
| Custom execute() logic      | Limited          | Full            | Operation-level   |
| Reusable as operation       | No               | No              | Yes               |
| Can use in YAML             | Yes              | No              | Yes               |
| Composable                  | With ops         | No              | Yes               |
| Use case                    | Code refactoring | One-off complex | Reusable backfill |

### Field Validators vs Model Validators

| Type                              | Use For                  | When Evaluated             | Access to Other Fields |
| --------------------------------- | ------------------------ | -------------------------- | ---------------------- |
| `@field_validator`                | Single field validation  | During field assignment    | No                     |
| `@model_validator(mode="before")` | Pre-validation transform | Before field validation    | Yes (raw data)         |
| `@model_validator(mode="after")`  | Cross-field validation   | After all fields validated | Yes (validated)        |

## Common Patterns

### 1. Dynamic Operations Based on Config

```python
def build_operations(self) -> list[dict[str, str]]:
    operations = []

    # Read from config file
    config = load_config()
    if config.get("enable_validation"):
        operations.append({"type": "..."})

    return operations
```

### 2. Loading from External Source

```python
def execute(self, store, project, *, dry_run=False):
    # Load from S3/database/API
    external_data = load_external_data()

    # Get field_provenance from Metaxy
    diff = store.resolve_update(feature_cls, samples=samples_df)

    # Join and write
    df = external_data.join(diff.added, on="sample_uid")
    store.write_metadata(feature_cls, df)
```

### 3. Validation with Clear Errors

```python
@field_validator("url")
@classmethod
def validate_url(cls, v: str) -> str:
    if not v.startswith("https://"):
        raise ValueError(f"URL must be HTTPS, got: {v}")
    return v


@model_validator(mode="after")
def validate_connectivity(self):
    # Check external resource is accessible
    if not check_connection(self.url):
        raise ValueError(f"Cannot connect to {self.url}")
    return self
```

### 4. Filtering Affected Features

```python
def get_affected_features(self, store, project):
    all_affected = super().get_affected_features(store, project)

    # Filter based on conditions
    graph = FeatureGraph.get_active()
    filtered = [key for key in all_affected if should_include(key, graph)]

    return filtered
```

## Best Practices

### Migration Design

1. **Start simple:** Use YAML if possible, Python when needed
2. **Validate early:** Check configuration and resources before expensive operations
3. **Support dry_run:** Always implement preview mode
4. **Handle errors:** Gracefully handle failures and return informative errors
5. **Use parent chain:** Link migrations for proper ordering

### Code Organization

1. **Reusable operations:** Extract common patterns into MetadataBackfill subclasses
2. **Clear naming:** Use descriptive migration IDs and class names
3. **Documentation:** Add docstrings explaining use case and key concepts
4. **Type hints:** Use comprehensive type annotations
5. **Validation:** Use Pydantic validators for configuration

### Testing

1. **Unit test validators:** Test field and model validators in isolation
2. **Test with fixtures:** Use sample data for testing backfill logic
3. **Mock external resources:** Use mocks for S3, databases, APIs in tests
4. **Test dry_run:** Ensure dry_run mode works correctly
5. **Test error cases:** Verify error handling and messages

### Execution

1. **Run in dev first:** Test migrations in development environment
2. **Check dry_run:** Always run with `--dry-run` first
3. **Monitor progress:** Use logging to track execution
4. **Verify results:** Check metadata after migration completes
5. **Be patient:** Large migrations can take time

## File Naming Convention

Python migration files should follow the pattern:

```
YYYYMMDD_HHMMSS_description.py
```

Examples:

- `20250101_120000_simple_reconciliation.py`
- `20250115_143000_backfill_s3_videos.py`
- `20250201_090000_database_migration.py`

The migration ID (defined in the class) should match the filename prefix for easy identification.

## Running Migrations

### List Available Migrations

```bash
metaxy migrations status
```

### Run Specific Migration (Dry Run)

```bash
metaxy migrations apply --dry-run 20250101_120000_simple_reconciliation
```

### Run All Pending Migrations

```bash
metaxy migrations apply
```

### Check Migration Status

```bash
metaxy migrations status
```

## Additional Resources

- **Main Documentation:** See `docs/migrations/` for detailed migration system documentation
- **Operation Reference:** See `src/metaxy/migrations/ops.py` for built-in operations
- **Migration Models:** See `src/metaxy/migrations/models.py` for base classes
- **Example Project:** See `examples/example-migration/` for a complete working example

## Troubleshooting

### Migration Not Found

- Ensure the file is in `.metaxy/migrations/` directory
- Check that the file defines exactly one Migration subclass
- Verify the migration_id matches expected format
- Check for syntax errors in the Python file

### Validation Errors

- Read the error message carefully - Pydantic provides detailed feedback
- Check field values match expected types and constraints
- Verify external resources are accessible
- Consider adding `require_validation=False` flags for development

### Import Errors

- Ensure all dependencies are installed (boto3, psycopg2, etc.)
- Check that custom operation classes are importable
- Verify type field in ops matches the actual class path
- Use try/except ImportError for optional dependencies

### Execution Failures

- Check logs for detailed error messages
- Verify feature graph is loaded correctly
- Ensure upstream features exist and have metadata
- Check that feature_key matches actual feature in graph
- Use dry_run mode to validate before executing

## Contributing

When adding new examples:

1. Follow the existing file naming and structure
2. Include comprehensive docstrings
3. Add clear comments explaining key concepts
4. Demonstrate a distinct use case
5. Update this README with a new section
6. Include usage examples and troubleshooting tips

## Questions?

For questions or issues:

1. Check the main documentation in `docs/migrations/`
2. Review existing examples in this directory
3. Look at tests in `tests/migrations/`
4. Open an issue on GitHub with details
